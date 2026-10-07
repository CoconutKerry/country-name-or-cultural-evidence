#include "ggml-backend.h"
#include "llama.h"
#include "json.hpp"

#include <algorithm>
#include <cctype>
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

std::string label_continuation(const std::string & label, const std::string & prompt) {
    if (prompt.empty()) {
        throw std::runtime_error("serialized prompt is empty");
    }
    const unsigned char last = static_cast<unsigned char>(prompt.back());
    return std::isspace(last) ? label : " " + label;
}

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t leading_token_count(
    const std::vector<llama_token> & tokens,
    llama_token token
) {
    size_t count = 0;
    if (token == LLAMA_TOKEN_NULL) {
        return count;
    }
    while (count < tokens.size() && tokens[count] == token) {
        ++count;
    }
    return count;
}

struct SerializationInspection {
    std::string serialized_prompt;
    std::string reference_serialized_prompt;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> prompt_tokens_add_special_false;
    std::vector<llama_token> prompt_tokens_add_special_true;
    std::vector<llama_token> reference_prompt_tokens;
    std::string selected_actual_tokenization_mode;
    bool selected_actual_add_special;
    llama_token bos_token;
    size_t leading_bos_token_count;
    size_t leading_bos_count_add_special_false;
    size_t leading_bos_count_add_special_true;
    size_t reference_leading_bos_token_count;
    json contract_checks;
    bool contract_pass;
};

SerializationInspection inspect_serialization(
    const llama_vocab * vocab,
    const char * chat_template,
    const json & request
) {
    if (!request.contains("reference_serialized_prompt") ||
        !request.at("reference_serialized_prompt").is_string()) {
        throw std::runtime_error("reference_serialized_prompt must be supplied");
    }
    const std::string serialized_prompt = apply_embedded_chat_template(
        chat_template,
        request.at("messages")
    );
    const std::string reference_serialized_prompt =
        request.at("reference_serialized_prompt").get<std::string>();
    if (serialized_prompt.empty() || reference_serialized_prompt.empty()) {
        throw std::runtime_error("serialized prompts must not be empty");
    }

    // The independent expansion explicitly contains <bos>, so its canonical
    // tokenization never adds specials. llama.cpp's formatter can render the
    // same template with or without a textual BOS; retain both actual paths.
    const std::vector<llama_token> prompt_tokens_without_special = tokenize(
        vocab, serialized_prompt, false
    );
    const std::vector<llama_token> prompt_tokens_with_special = tokenize(
        vocab, serialized_prompt, true
    );
    const std::vector<llama_token> reference_prompt_tokens = tokenize(
        vocab, reference_serialized_prompt, false
    );
    if (prompt_tokens_without_special.empty() ||
        prompt_tokens_with_special.empty() || reference_prompt_tokens.empty()) {
        throw std::runtime_error("serialized prompt tokenized to an empty sequence");
    }
    const llama_token bos_token = llama_vocab_bos(vocab);
    const size_t bos_count_without_special = leading_token_count(
        prompt_tokens_without_special, bos_token
    );
    const size_t bos_count_with_special = leading_token_count(
        prompt_tokens_with_special, bos_token
    );
    const size_t reference_bos_count = leading_token_count(
        reference_prompt_tokens, bos_token
    );

    const json contract = request.value("serialization_contract", json::object());
    const bool allow_token_equivalence = contract.value(
        "allow_token_id_equivalence", false
    );
    size_t required_bos_count = 0;
    bool has_required_bos_count = false;
    if (contract.contains("required_leading_bos_token_count") &&
        !contract.at("required_leading_bos_token_count").is_null()) {
        required_bos_count = contract.at(
            "required_leading_bos_token_count"
        ).get<size_t>();
        has_required_bos_count = true;
    }
    const bool select_actual_tokenization = contract.value(
        "select_actual_tokenization_mode", false
    );
    const bool without_special_is_safe =
        prompt_tokens_without_special == reference_prompt_tokens &&
        (!has_required_bos_count ||
         bos_count_without_special == required_bos_count);
    const bool with_special_is_safe =
        prompt_tokens_with_special == reference_prompt_tokens &&
        (!has_required_bos_count || bos_count_with_special == required_bos_count);

    // Deterministic and fail-closed: prefer the no-added-specials path when it
    // is already canonical; otherwise use add_special=true only when that
    // exactly restores the canonical token IDs and required BOS count.
    std::vector<llama_token> prompt_tokens;
    std::string selected_mode;
    bool selected_add_special = false;
    bool tokenization_mode_selected = true;
    if (!select_actual_tokenization || without_special_is_safe) {
        prompt_tokens = prompt_tokens_without_special;
        selected_mode = "add_special_false";
    } else if (with_special_is_safe) {
        prompt_tokens = prompt_tokens_with_special;
        selected_mode = "add_special_true";
        selected_add_special = true;
    } else {
        tokenization_mode_selected = false;
    }
    const size_t bos_count = leading_token_count(prompt_tokens, bos_token);
    const bool byte_exact = serialized_prompt == reference_serialized_prompt;
    const bool token_ids_equal = tokenization_mode_selected &&
        prompt_tokens == reference_prompt_tokens;
    const bool equivalence = byte_exact ||
        (allow_token_equivalence && token_ids_equal);

    bool suffix_verified = true;
    if (contract.contains("required_suffix") &&
        !contract.at("required_suffix").is_null()) {
        const std::string suffix = contract.at("required_suffix").get<std::string>();
        suffix_verified = !suffix.empty() &&
            has_suffix(serialized_prompt, suffix) &&
            has_suffix(reference_serialized_prompt, suffix);
    }
    const bool bos_verified = !has_required_bos_count ||
        (tokenization_mode_selected && bos_count == required_bos_count &&
         reference_bos_count == required_bos_count);
    const json checks = {
        {"byte_exact", byte_exact},
        {"token_ids_equal", token_ids_equal},
        {"serialization_equivalence", equivalence},
        {"required_suffix", suffix_verified},
        {"required_leading_bos_token_count", bos_verified},
        {"actual_tokenization_mode_selected", tokenization_mode_selected},
        {"actual_add_special_false_safe", without_special_is_safe},
        {"actual_add_special_true_safe", with_special_is_safe},
    };
    return {
        serialized_prompt,
        reference_serialized_prompt,
        prompt_tokens,
        prompt_tokens_without_special,
        prompt_tokens_with_special,
        reference_prompt_tokens,
        selected_mode,
        selected_add_special,
        bos_token,
        bos_count,
        bos_count_without_special,
        bos_count_with_special,
        reference_bos_count,
        checks,
        tokenization_mode_selected && equivalence && suffix_verified && bos_verified,
    };
}

json serialization_response(
    const llama_vocab * vocab,
    const char * chat_template,
    const json & request
) {
    const SerializationInspection inspection = inspect_serialization(
        vocab, chat_template, request
    );
    return {
        {"type", "serialization"},
        {"request_id", request.at("request_id")},
        {"structured_messages", request.at("messages")},
        {"serialized_prompt", inspection.serialized_prompt},
        {"reference_serialized_prompt", inspection.reference_serialized_prompt},
        {"prompt_token_ids", inspection.prompt_tokens},
        {"prompt_token_ids_add_special_false", inspection.prompt_tokens_add_special_false},
        {"prompt_token_ids_add_special_true", inspection.prompt_tokens_add_special_true},
        {"reference_prompt_token_ids", inspection.reference_prompt_tokens},
        {"selected_actual_tokenization_mode", inspection.selected_actual_tokenization_mode},
        {"prompt_token_count", inspection.prompt_tokens.size()},
        {"reference_prompt_token_count", inspection.reference_prompt_tokens.size()},
        {"bos_token_id", inspection.bos_token},
        {"leading_bos_token_count", inspection.leading_bos_token_count},
        {"leading_bos_count_add_special_false", inspection.leading_bos_count_add_special_false},
        {"leading_bos_count_add_special_true", inspection.leading_bos_count_add_special_true},
        {"reference_leading_bos_token_count", inspection.reference_leading_bos_token_count},
        {"serialization_contract_checks", inspection.contract_checks},
        {"serialization_contract_pass", inspection.contract_pass},
        {"inference_performed", false},
    };
}

json selected_model_metadata(const llama_model * model) {
    static const std::vector<std::string> keys = {
        "general.architecture",
        "general.name",
        "general.basename",
        "general.finetune",
        "general.size_label",
        "general.file_type",
        "general.quantization_version",
        "gemma2.context_length",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    };
    json metadata = json::object();
    for (const auto & key : keys) {
        const int32_t needed = llama_model_meta_val_str(model, key.c_str(), nullptr, 0);
        if (needed <= 0) {
            continue;
        }
        std::vector<char> buffer(static_cast<size_t>(needed) + 1U, '\0');
        const int32_t actual = llama_model_meta_val_str(
            model, key.c_str(), buffer.data(), buffer.size()
        );
        if (actual > 0) {
            metadata[key] = std::string(buffer.data(), static_cast<size_t>(actual));
        }
    }
    return metadata;
}

json score_request(
    llama_context * context,
    const llama_vocab * vocab,
    const char * chat_template,
    int32_t n_vocab,
    uint32_t n_ctx,
    const json & request
) {
    const std::string request_id = request.at("request_id").get<std::string>();
    const auto labels = request.at("labels").get<std::vector<std::string>>();
    if (labels.empty()) {
        throw std::runtime_error("labels must not be empty");
    }
    std::unordered_set<std::string> unique_labels;
    for (const auto & label : labels) {
        if (label.empty() || !unique_labels.insert(label).second) {
            throw std::runtime_error("candidate labels must be non-empty and unique");
        }
    }

    const SerializationInspection inspection = inspect_serialization(
        vocab, chat_template, request
    );
    // Enforce the formatter contract before llama_decode.  The separate
    // serialize-only request returns the same evidence even when this check
    // fails, so the Python smoke runner can persist a useful diagnostic.
    if (!inspection.contract_pass) {
        throw std::runtime_error(
            "serialization contract failed before inference"
        );
    }
    const std::string & serialized_prompt = inspection.serialized_prompt;
    const std::vector<llama_token> & prompt_tokens = inspection.prompt_tokens;
    if (prompt_tokens.size() >= n_ctx) {
        throw std::runtime_error("serialized prompt exceeds the configured context");
    }
    const llama_token bos_token = inspection.bos_token;
    const size_t leading_bos_count = inspection.leading_bos_token_count;

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
        const std::string continuation = label_continuation(label, serialized_prompt);
        const std::vector<llama_token> combined_tokens = tokenize(
            vocab,
            serialized_prompt + continuation,
            inspection.selected_actual_add_special
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

    return {
        {"type", "result"},
        {"request_id", request_id},
        {"structured_messages", request.at("messages")},
        {"serialized_prompt", serialized_prompt},
        {"reference_serialized_prompt", inspection.reference_serialized_prompt},
        {"prompt_token_ids", prompt_tokens},
        {"prompt_token_ids_add_special_false", inspection.prompt_tokens_add_special_false},
        {"prompt_token_ids_add_special_true", inspection.prompt_tokens_add_special_true},
        {"reference_prompt_token_ids", inspection.reference_prompt_tokens},
        {"selected_actual_tokenization_mode", inspection.selected_actual_tokenization_mode},
        {"prompt_token_count", prompt_tokens.size()},
        {"reference_prompt_token_count", inspection.reference_prompt_tokens.size()},
        {"bos_token_id", bos_token},
        {"leading_bos_token_count", leading_bos_count},
        {"leading_bos_count_add_special_false", inspection.leading_bos_count_add_special_false},
        {"leading_bos_count_add_special_true", inspection.leading_bos_count_add_special_true},
        {"reference_leading_bos_token_count", inspection.reference_leading_bos_token_count},
        {"serialization_contract_checks", inspection.contract_checks},
        {"serialization_contract_pass", inspection.contract_pass},
        {"answer_target_position", prompt_tokens.size()},
        {"answer_predictive_logit_position", prompt_tokens.size() - 1U},
        {"prompt_prefix_verified", true},
        {"candidates", candidates},
        {"generated_answer", nullptr},
    };
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
            {"selected_model_metadata", selected_model_metadata(model.get())},
            {"vocabulary_size", llama_vocab_n_tokens(vocab)},
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
                const std::string operation = request.value(
                    "operation", std::string("score")
                );
                if (operation == "serialize_only") {
                    std::cout << serialization_response(
                        vocab,
                        chat_template,
                        request
                    ).dump() << std::endl;
                } else if (operation == "score") {
                    std::cout << score_request(
                        context.get(),
                        vocab,
                        chat_template,
                        llama_vocab_n_tokens(vocab),
                        llama_n_ctx(context.get()),
                        request
                    ).dump() << std::endl;
                } else {
                    throw std::runtime_error("unsupported helper operation");
                }
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
