// rex_attn_capture — Tier-2 attention capture from llama.cpp, strategy A (the stock cb_eval
// node callback, NO ggml patch). Loads a GGUF model, runs one forward pass over a prompt with
// Flash-Attention DISABLED (so the softmax(QKᵀ) tensor materializes), captures each layer's
// "kq_soft_max" tensor via the eval callback, averages over heads, and prints the per-layer
// [n_q × n_kv] attention as JSON on stdout. The rexgraph side (agent/agent/attn_introspect.py)
// feeds each layer's matrix to the RCF analyzer.
//
// This exposes what the OpenAI API never surfaces — the internal attention weights — to the
// relational-complex math, on the Vulkan/llama.cpp stack. Build: see native/build.sh.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct LayerAttn { long long n_kv, n_q, n_head; std::vector<float> data; };
struct Capture { std::vector<LayerAttn> layers; };

// The eval callback fires per graph node, twice: ask=true (return true to request the data),
// then ask=false (data is ready). We want the post-softmax attention weights, named
// "kq_soft_max" in llama-graph.cpp.
static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (t == nullptr || t->name == nullptr) return true;
    const bool is_attn = std::strstr(t->name, "kq_soft_max") != nullptr;
    if (ask) return is_attn;
    if (is_attn && t->type == GGML_TYPE_F32) {
        Capture * cap = static_cast<Capture*>(user_data);
        LayerAttn la;
        la.n_kv = t->ne[0]; la.n_q = t->ne[1]; la.n_head = t->ne[2] ? t->ne[2] : 1;
        const size_t n = ggml_nelements(t);
        la.data.resize(n);
        ggml_backend_tensor_get(t, la.data.data(), 0, n * sizeof(float));   // works CPU + GPU
        cap->layers.push_back(std::move(la));
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf \"prompt\" [n_gpu_layers=999]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const std::string prompt = argv[2];
    const int ngl = argc > 3 ? std::atoi(argv[3]) : 999;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), /*add_special*/ true, /*parse_special*/ false);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, false); }
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    Capture cap;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 8; cp.n_batch = n + 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;      // REQUIRED: else softmax is fused away
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) {
        std::fprintf(stderr, "decode failed\n"); return 1;
    }

    // per layer: average over heads -> [n_q][n_kv], emit JSON
    std::printf("{\"n_tokens\":%d,\"n_layers\":%zu,\"layers\":[", n, cap.layers.size());
    for (size_t li = 0; li < cap.layers.size(); ++li) {
        const LayerAttn & la = cap.layers[li];
        const long long nk = la.n_kv, nq = la.n_q, nh = la.n_head;
        std::printf("%s{\"layer\":%zu,\"n_kv\":%lld,\"n_q\":%lld,\"n_head\":%lld,\"attn\":[",
                    li ? "," : "", li, nk, nq, nh);
        for (long long q = 0; q < nq; ++q) {
            std::printf("%s[", q ? "," : "");
            for (long long k = 0; k < nk; ++k) {
                double s = 0.0;
                for (long long h = 0; h < nh; ++h) s += la.data[(size_t)(k + q * nk + h * nk * nq)];
                std::printf("%s%.5g", k ? "," : "", s / (double)nh);
            }
            std::printf("]");
        }
        std::printf("]}");
    }
    std::printf("]}\n");

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
