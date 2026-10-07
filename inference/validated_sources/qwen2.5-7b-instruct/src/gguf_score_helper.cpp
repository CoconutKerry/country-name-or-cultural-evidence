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

    const std::string serialized_prompt = apply_embedded_chat_template(
        chat_template,
        request.at("messages")
    );
    const std::vector<llama_token> prompt_tokens = tokenize(vocab, serialized_prompt, true);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("serialized prompt tokenized to an empty sequence");
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
        const std::string continuation = label_continuation(label, serialized_prompt);
        const std::vector<llama_token> combined_tokens = tokenize(
            vocab,
            serialized_prompt + continuation,
            true
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
        {"prompt_token_ids", prompt_tokens},
        {"prompt_token_count", prompt_tokens.size()},
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
                std::cout << score_request(
                    context.get(),
                    vocab,
                    chat_template,
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
