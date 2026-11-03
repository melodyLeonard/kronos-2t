package com.kronos.mobile

import java.io.File

fun interface PieceSink {
    fun onPiece(utf8: ByteArray): Boolean
}

/** Binds to `src/ffi/android_jni.rs`. */
object KronosEngine {
    init {
        System.loadLibrary("kronos_2t")
    }

    @JvmStatic external fun nativeCreate(dir: ByteArray, mode: ByteArray, ctx: Int, threads: Int, budgetMb: Int): Long
    @JvmStatic external fun nativeCreateError(): ByteArray
    @JvmStatic external fun nativeConfigureKv(handle: Long, bits: Int, window: Int, sink: Int): Int
    @JvmStatic external fun nativeGenerate(handle: Long, messagesJson: ByteArray, maxTokens: Int, temperature: Float, sink: PieceSink, metrics: DoubleArray): Int
    @JvmStatic external fun nativeLastError(handle: Long): ByteArray
    @JvmStatic external fun nativeReset(handle: Long)
    @JvmStatic external fun nativeDestroy(handle: Long)
    @JvmStatic external fun nativePeakRss(): Long

    /** Performance cores only: little cores stall the shared-work thread pool. */
    fun performanceCores(): Int {
        val all = Runtime.getRuntime().availableProcessors()
        val freqs = (0 until all).mapNotNull {
            runCatching { File("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
        }
        val fast = freqs.maxOrNull()?.let { top -> freqs.count { it >= top * 7 / 10 } } ?: 0
        return (if (fast in 1..all) fast else all).coerceIn(1, 4)
    }
}
