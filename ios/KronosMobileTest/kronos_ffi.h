#ifndef KRONOS_FFI_H
#define KRONOS_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KronosMobileSession KronosMobileSession;

typedef bool (*KronosStreamCallback)(const char *utf8_ptr, size_t utf8_len, void *user_data);

KronosMobileSession *kronos_session_create(
    const char *model_dir_utf8,
    const char *mode_utf8,
    size_t ctx_tokens,
    size_t threads,
    size_t memory_budget_mb
);

void kronos_session_destroy(KronosMobileSession *session);

const char *kronos_session_last_error(const KronosMobileSession *session);

int32_t kronos_session_enable_metal(KronosMobileSession *session);

int32_t kronos_session_disable_metal(KronosMobileSession *session);

int32_t kronos_session_configure_kv(
    KronosMobileSession *session,
    uint8_t kv_bits,
    size_t sliding_window,
    size_t sink_tokens
);

int32_t kronos_session_attach_draft(
    KronosMobileSession *session,
    const char *draft_dir_utf8,
    const char *draft_mode_utf8
);

int32_t kronos_session_jev_decide(
    KronosMobileSession *session,
    const char *system_utf8,
    const char *query_utf8,
    const char *options_json_utf8,
    float temperature,
    float *out_probs_f32,
    size_t max_options,
    float *out_entropy_f32,
    float *out_confidence_f32
);

int32_t kronos_session_prefill(
    KronosMobileSession *session,
    const char *system_utf8,
    const char *user_utf8
);

int32_t kronos_session_step(
    KronosMobileSession *session,
    char *out_utf8_buf,
    size_t out_buf_cap,
    int32_t *out_token_id
);

int32_t kronos_session_generate(
    KronosMobileSession *session,
    const char *system_utf8,
    const char *user_utf8,
    size_t max_tokens,
    float temperature,
    KronosStreamCallback on_piece_cb,
    void *user_data
);

int32_t kronos_session_generate_chat_json(
    KronosMobileSession *session,
    const char *messages_json_utf8,
    size_t max_tokens,
    float temperature,
    KronosStreamCallback on_piece_cb,
    void *user_data,
    double *out_prefill_ms,
    double *out_decode_ms,
    size_t *out_prefilled_tokens
);

void kronos_session_reset(KronosMobileSession *session);

int32_t kronos_ingest_attachment_bytes(
    const char *filename_utf8,
    const char *mime_utf8,
    const uint8_t *bytes_ptr,
    size_t bytes_len,
    const char *workdir_utf8,
    size_t slot,
    char *out_badge_utf8,
    size_t out_badge_cap,
    char *out_context_utf8,
    size_t out_context_cap,
    char *out_saved_rel_utf8,
    size_t out_saved_rel_cap
);

int32_t kronos_session_run_agent_turn(
    KronosMobileSession *session,
    const char *workdir_utf8,
    const char *messages_json_utf8,
    size_t max_steps,
    size_t max_tokens,
    float temperature,
    KronosStreamCallback on_piece_cb,
    void *user_data,
    char *out_final_utf8,
    size_t out_final_cap
);

size_t kronos_session_resident_bytes(const KronosMobileSession *session);

size_t kronos_session_process_rss_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
