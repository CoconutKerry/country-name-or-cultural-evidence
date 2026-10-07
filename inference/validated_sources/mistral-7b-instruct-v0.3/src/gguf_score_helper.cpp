#include "ggml-backend.h"
#include "llama.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

struct ModelDeleter {
    void operator()(llama_model * value) const {
        if (value != nullptr) {
            llama_model_free(value);
        }
    }
};

struct ContextDeleter {
    void operator()(llama_context * value) const {
        if (value != nullptr) {
            llama_free(value);
        }
    }
};

std::vector<llama_token> tokenize(
    const llama_vocab * vocab,
    const std::string & text,
    bool add_special
) {
    int32_t needed = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        true
    );
    if (needed == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization length overflow");
    }
    if (needed < 0) {
        needed = -needed;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    const int32_t actual = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        true
    );
    if (actual < 0) {
        throw std::runtime_error("tokenization buffer unexpectedly too small");
    }
    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}

std::string apply_embedded_chat_template(
    const char * chat_template,
    const json & messages
) {
    if (chat_template == nullptr || *chat_template == '\0') {
        throw std::runtime_error("the GGUF has no embedded chat template");
    }
    if (!messages.is_array() || messages.empty()) {
        throw std::runtime_error("messages must be a non-empty array");
    }

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(messages.size());
    contents.reserve(messages.size());
    for (const auto & message : messages) {
        if (!message.is_object() || !message.contains("role") || !message.contains("content")) {
            throw std::runtime_error("each message needs role and content strings");
        }
        roles.push_back(message.at("role").get<std::string>());
        contents.push_back(message.at("content").get<std::string>());
        if (roles.back().empty() || contents.back().empty()) {
            throw std::runtime_error("message role and content must be non-empty");
        }
    }

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (size_t index = 0; index < roles.size(); ++index) {
        chat.push_back({roles[index].c_str(), contents[index].c_str()});
    }

    int32_t needed = llama_chat_apply_template(
        chat_template,
        chat.data(),
        chat.size(),
        true,
        nullptr,
        0
    );
    if (needed <= 0) {
        throw std::runtime_error("embedded chat template could not be applied");
    }
    std::vector<char> buffer(static_cast<size_t>(needed) + 1U, '\0');
    const int32_t actual = llama_chat_apply_template(
        chat_template,
        chat.data(),
        chat.size(),
        true,
        buffer.data(),
        static_cast<int32_t>(buffer.size())
    );
    if (actual <= 0 || actual > needed) {
        throw std::runtime_error("embedded chat template returned an invalid length");
    }
    return std::string(buffer.data(), static_cast<size_t>(actual));
}

std::pair<double, double> target_logit_and_log_probability(
    const float * logits,
    int32_t n_vocab,
    llama_token target
) {
    if (logits == nullptr) {
        throw std::runtime_error("llama_get_logits_ith returned null");
    }
    if (target < 0 || target >= n_vocab) {
        throw std::runtime_error("candidate token ID is outside the vocabulary");
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (int32_t index = 0; index < n_vocab; ++index) {
        maximum = std::max(maximum, static_cast<double>(logits[index]));
    }
    double denominator = 0.0;
    for (int32_t index = 0; index < n_vocab; ++index) {
        denominator += std::exp(static_cast<double>(logits[index]) - maximum);
    }
    const double raw_logit = static_cast<double>(logits[target]);
    const double log_probability = raw_logit - maximum - std::log(denominator);
    if (!std::isfinite(raw_logit) || !std::isfinite(log_probability)) {
        throw std::runtime_error("candidate score is non-finite");
    }
    return {raw_logit, log_probability};
}

json score_request(
    llama_context * context,
    const llama_vocab * vocab,
    const char * chat_template,
    llama_token bos_token,
    int32_t n_vocab,
    uint32_t n_ctx,
    const json & request
) {
    const std::string request_id = request.at("request_id").get<std::string>();
    const auto labels = request.at("labels").get<std::vector<std::string>>();
    if (labels.empty()) {
        throw std::runtime_error("labels must not be empty");
    }
    if (!request.contains("candidate_continuations") ||
        !request.at("candidate_continuations").is_object()) {
        throw std::runtime_error("candidate_continuations must be an object");
    }
    const json & candidate_continuations = request.at("candidate_continuations");
    if (!request.contains("add_special_tokens") ||
        !request.at("add_special_tokens").is_boolean() ||
        !request.contains("tokenization_add_special") ||
        !request.at("tokenization_add_special").is_boolean()) {
        throw std::runtime_error("tokenization special-token mode must be explicit");
    }
    const bool add_special_tokens = request.at("add_special_tokens").get<bool>();
    if (request.at("tokenization_add_special").get<bool>() != add_special_tokens) {
        throw std::runtime_error("tokenization special-token mode aliases disagree");
    }
    if (!request.contains("candidate_prefix") ||
        !request.at("candidate_prefix").is_string()) {
        throw std::runtime_error("candidate_prefix must be explicit");
    }
    const std::string candidate_prefix = request.at("candidate_prefix").get<std::string>();
    std::unordered_set<std::string> unique_labels;
    for (const auto & label : labels) {
        if (label.empty() || !unique_labels.insert(label).second) {
            throw std::runtime_error("candidate labels must be non-empty and unique");
        }
        if (!candidate_continuations.contains(label) ||
            !candidate_continuations.at(label).is_string() ||
            candidate_continuations.at(label).get<std::string>().empty()) {
            throw std::runtime_error("every label needs a non-empty explicit continuation");
        }
    }

    const std::string serialized_prompt = apply_embedded_chat_template(
        chat_template,
        request.at("messages")
    );
    const std::vector<llama_token> prompt_tokens = tokenize(
        vocab,
        serialized_prompt,
        add_special_tokens
    );
    if (prompt_tokens.empty()) {
        throw std::runtime_error("serialized prompt tokenized to an empty sequence");
    }
    const size_t prompt_bos_token_count = static_cast<size_t>(std::count(
        prompt_tokens.begin(), prompt_tokens.end(), bos_token
    ));
    const bool prompt_starts_with_bos = prompt_tokens.front() == bos_token;
    const bool compare_to_canonical = request.contains("canonical_serialized_prompt");
    std::string canonical_serialized_prompt;
    std::vector<llama_token> canonical_prompt_tokens;
    size_t canonical_prompt_bos_token_count = 0U;
    bool canonical_prompt_starts_with_bos = false;
    bool template_equivalence_preflight_passed = false;
    std::string template_equivalence_policy;
    if (compare_to_canonical) {
        if (!request.at("canonical_serialized_prompt").is_string() ||
            !request.contains("canonical_tokenization_add_special") ||
            !request.at("canonical_tokenization_add_special").is_boolean() ||
            request.at("canonical_tokenization_add_special").get<bool>() ||
            !request.contains("template_equivalence_policy") ||
            !request.at("template_equivalence_policy").is_string() ||
            !request.contains("chat_template_source") ||
            request.at("chat_template_source") != "embedded_gguf_metadata") {
            throw std::runtime_error(
                "canonical Mistral template comparison contract is incomplete"
            );
        }
        template_equivalence_policy = request.at(
            "template_equivalence_policy"
        ).get<std::string>();
        if (template_equivalence_policy !=
            "embedded_serialization_token_ids_equal_canonical_with_one_leading_bos") {
            throw std::runtime_error("unsupported Mistral template-equivalence policy");
        }
        canonical_serialized_prompt = request.at(
            "canonical_serialized_prompt"
        ).get<std::string>();
        if (canonical_serialized_prompt.empty()) {
            throw std::runtime_error("canonical Mistral serialization must not be empty");
        }
        canonical_prompt_tokens = tokenize(
            vocab,
            canonical_serialized_prompt,
            false
        );
        if (canonical_prompt_tokens.empty()) {
            throw std::runtime_error(
                "canonical Mistral serialization tokenized to an empty sequence"
            );
        }
        canonical_prompt_bos_token_count = static_cast<size_t>(std::count(
            canonical_prompt_tokens.begin(), canonical_prompt_tokens.end(), bos_token
        ));
        canonical_prompt_starts_with_bos =
            canonical_prompt_tokens.front() == bos_token;
        template_equivalence_preflight_passed =
            add_special_tokens &&
            prompt_starts_with_bos &&
            prompt_bos_token_count == 1U &&
            canonical_prompt_starts_with_bos &&
            canonical_prompt_bos_token_count == 1U &&
            prompt_tokens == canonical_prompt_tokens;
        if (!template_equivalence_preflight_passed) {
            return {
                {"type", "template_validation_error"},
                {"request_id", request_id},
                {"message", "embedded Mistral serialization failed canonical token-ID/BOS equivalence"},
                {"structured_messages", request.at("messages")},
                {"serialized_prompt", serialized_prompt},
                {"canonical_serialized_prompt", canonical_serialized_prompt},
                {"prompt_token_ids", prompt_tokens},
                {"canonical_prompt_token_ids", canonical_prompt_tokens},
                {"prompt_token_count", prompt_tokens.size()},
                {"canonical_prompt_token_count", canonical_prompt_tokens.size()},
                {"bos_token_id", bos_token},
                {"prompt_bos_token_count", prompt_bos_token_count},
                {"canonical_prompt_bos_token_count", canonical_prompt_bos_token_count},
                {"prompt_starts_with_bos", prompt_starts_with_bos},
                {"canonical_prompt_starts_with_bos", canonical_prompt_starts_with_bos},
                {"add_special_tokens", add_special_tokens},
                {"tokenization_add_special", add_special_tokens},
                {"canonical_tokenization_add_special", false},
                {"candidate_prefix", candidate_prefix},
                {"chat_template_source", "embedded_gguf_metadata"},
                {"template_equivalence_policy", template_equivalence_policy},
                {"template_equivalence_preflight_passed", false},
                {"generated_answer", nullptr},
            };
        }
    }
    if (prompt_tokens.size() >= n_ctx) {
        throw std::runtime_error("serialized prompt exceeds the configured context");
    }

    llama_memory_t memory = llama_get_memory(context);
    llama_memory_clear(memory, false);
    std::vector<llama_token> mutable_prompt = prompt_tokens;
    llama_batch prompt_batch = llama_batch_get_one(
        mutable_prompt.data(),
        static_cast<int32_t>(mutable_prompt.size())
    );
    const int32_t prompt_status = llama_decode(context, prompt_batch);
    if (prompt_status != 0) {
        throw std::runtime_error("llama_decode failed while evaluating the prompt");
    }
    const float * prompt_logits_ptr = llama_get_logits_ith(context, -1);
    if (prompt_logits_ptr == nullptr) {
        throw std::runtime_error("prompt answer-position logits are unavailable");
    }
    std::vector<float> prompt_logits(
        prompt_logits_ptr,
        prompt_logits_ptr + static_cast<size_t>(n_vocab)
    );

    json candidates = json::object();
    for (const auto & label : labels) {
        const std::string continuation = candidate_continuations.at(label).get<std::string>();
        const std::vector<llama_token> combined_tokens = tokenize(
            vocab,
            serialized_prompt + continuation,
            add_special_tokens
        );
        if (combined_tokens.size() < prompt_tokens.size() ||
            !std::equal(prompt_tokens.begin(), prompt_tokens.end(), combined_tokens.begin())) {
            throw std::runtime_error(
                "prompt tokenization is not a prefix of prompt-plus-label tokenization"
            );
        }
        std::vector<llama_token> continuation_tokens(
            combined_tokens.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
            combined_tokens.end()
        );
        if (continuation_tokens.empty()) {
            throw std::runtime_error("a candidate label tokenized to an empty continuation");
        }
        if (combined_tokens.size() >= n_ctx) {
            throw std::runtime_error("prompt-plus-label exceeds the configured context");
        }

        json raw_logits = json::array();
        json token_log_probabilities = json::array();
        json target_positions = json::array();
        json predictive_positions = json::array();
        double sequence_log_probability = 0.0;

        for (size_t offset = 0; offset < continuation_tokens.size(); ++offset) {
            const int32_t target_position = static_cast<int32_t>(prompt_tokens.size() + offset);
            const int32_t predictive_position = target_position - 1;
            const float * logits = nullptr;
            if (offset == 0) {
                logits = prompt_logits.data();
            } else {
                llama_token previous = continuation_tokens[offset - 1];
                llama_batch token_batch = llama_batch_get_one(&previous, 1);
                const int32_t token_status = llama_decode(context, token_batch);
                if (token_status != 0) {
                    throw std::runtime_error("llama_decode failed within a candidate sequence");
                }
                logits = llama_get_logits_ith(context, -1);
            }
            const auto [raw_logit, token_log_probability] =
                target_logit_and_log_probability(
                    logits,
                    n_vocab,
                    continuation_tokens[offset]
                );
            raw_logits.push_back(raw_logit);
            token_log_probabilities.push_back(token_log_probability);
            target_positions.push_back(target_position);
            predictive_positions.push_back(predictive_position);
            sequence_log_probability += token_log_probability;
        }

        const bool removed = llama_memory_seq_rm(
            memory,
            0,
            static_cast<llama_pos>(prompt_tokens.size()),
            -1
        );
        if (!removed) {
            throw std::runtime_error("could not roll back candidate tokens from model memory");
        }
        candidates[label] = {
            {"candidate_label", label},
            {"continuation", continuation},
            {"token_ids", continuation_tokens},
            {"token_count", continuation_tokens.size()},
            {"raw_token_logits", raw_logits},
            {"token_log_probabilities", token_log_probabilities},
            {"sequence_log_probability", sequence_log_probability},
            {"target_token_positions", target_positions},
            {"predictive_logit_positions", predictive_positions},
            {"prompt_prefix_verified", true},
        };
    }

    json result = {
        {"type", "result"},
        {"request_id", request_id},
        {"structured_messages", request.at("messages")},
        {"serialized_prompt", serialized_prompt},
        {"prompt_token_ids", prompt_tokens},
        {"prompt_token_count", prompt_tokens.size()},
        {"bos_token_id", bos_token},
        {"prompt_bos_token_count", prompt_bos_token_count},
        {"prompt_starts_with_bos", prompt_starts_with_bos},
        {"answer_target_position", prompt_tokens.size()},
        {"answer_predictive_logit_position", prompt_tokens.size() - 1U},
        {"add_special_tokens", add_special_tokens},
        {"tokenization_add_special", add_special_tokens},
        {"candidate_prefix", candidate_prefix},
        {"prompt_prefix_verified", true},
        {"candidates", candidates},
        {"generated_answer", nullptr},
    };
    if (compare_to_canonical) {
        result["canonical_serialized_prompt"] = canonical_serialized_prompt;
        result["canonical_prompt_token_ids"] = canonical_prompt_tokens;
        result["canonical_prompt_token_count"] = canonical_prompt_tokens.size();
        result["canonical_prompt_bos_token_count"] = canonical_prompt_bos_token_count;
        result["canonical_prompt_starts_with_bos"] = canonical_prompt_starts_with_bos;
        result["canonical_tokenization_add_special"] = false;
        result["chat_template_source"] = "embedded_gguf_metadata";
        result["template_equivalence_policy"] = template_equivalence_policy;
        result["template_equivalence_preflight_passed"] =
            template_equivalence_preflight_passed;
    }
    return result;
}

int parse_positive_int(const char * value, const char * name) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 7 || std::string(argv[1]) != "--model" ||
        std::string(argv[3]) != "--threads" || std::string(argv[5]) != "--ctx-size") {
        std::cerr << "usage: gguf_score_helper --model MODEL --threads N --ctx-size N\n";
        return 2;
    }

    try {
        const std::string model_path = argv[2];
        const int threads = parse_positive_int(argv[4], "threads");
        const int context_size = parse_positive_int(argv[6], "ctx-size");

        llama_backend_init();
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, ModelDeleter> model(
            llama_model_load_from_file(model_path.c_str(), model_params)
        );
        if (!model) {
            throw std::runtime_error("failed to load GGUF model");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        if (vocab == nullptr) {
            throw std::runtime_error("loaded model has no vocabulary");
        }
        const char * chat_template = llama_model_chat_template(model.get(), nullptr);
        if (chat_template == nullptr || *chat_template == '\0') {
            throw std::runtime_error("official GGUF is missing its embedded chat template");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(context_size);
        context_params.n_batch = static_cast<uint32_t>(context_size);
        context_params.n_ubatch = std::min<uint32_t>(512U, context_params.n_batch);
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.embeddings = false;
        context_params.no_perf = false;
        std::unique_ptr<llama_context, ContextDeleter> context(
            llama_init_from_model(model.get(), context_params)
        );
        if (!context) {
            throw std::runtime_error("failed to create llama.cpp context");
        }
        llama_set_n_threads(context.get(), threads, threads);

        char description[512] = {};
        llama_model_desc(model.get(), description, sizeof(description));
        json ready = {
            {"type", "ready"},
            {"backend", "llama.cpp"},
            {"llama_version", llama_version()},
            {"llama_system_info", llama_print_system_info()},
            {"model_description", description},
            {"loaded_model_size_bytes", llama_model_size(model.get())},
            {"model_parameter_count", llama_model_n_params(model.get())},
            {"vocabulary_size", llama_vocab_n_tokens(vocab)},
            {"bos_token_id", llama_vocab_bos(vocab)},
            {"requested_context_size", context_size},
            {"actual_context_size", llama_n_ctx(context.get())},
            {"threads", threads},
            {"chat_template", std::string(chat_template)},
            {"chat_template_source", "embedded_gguf_metadata"},
            {"gpu_layers", 0},
        };
        std::cout << ready.dump() << std::endl;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const json request = json::parse(line);
                std::cout << score_request(
                    context.get(),
                    vocab,
                    chat_template,
                    llama_vocab_bos(vocab),
                    llama_vocab_n_tokens(vocab),
                    llama_n_ctx(context.get()),
                    request
                ).dump() << std::endl;
            } catch (const std::exception & error) {
                json failure = {
                    {"type", "error"},
                    {"message", error.what()},
                };
                std::cout << failure.dump() << std::endl;
                return 1;
            }
        }

        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "GGUF scorer fatal error: " << error.what() << '\n';
        return 1;
    }
}
