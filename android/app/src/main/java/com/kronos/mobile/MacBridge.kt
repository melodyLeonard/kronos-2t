package com.kronos.mobile

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.TimeZone

/** Client for `kronos_device_bridge` on the paired Mac. */
class MacBridge(baseUrl: String, private val token: String) {
    private val base = baseUrl.trim().trimEnd('/').let { if (it.contains("://")) it else "http://$it" }

    private fun open(path: String, timeoutMs: Int): HttpURLConnection {
        val c = URL(base + path).openConnection() as HttpURLConnection
        c.connectTimeout = 4000
        c.readTimeout = timeoutMs
        c.setRequestProperty("Authorization", "Bearer $token")
        return c
    }

    private fun finish(c: HttpURLConnection): String {
        val code = c.responseCode
        if (code == 401) throw IOException("The Mac rejected this token.")
        if (code !in 200..299) throw IOException("The Mac answered HTTP $code.")
        return c.inputStream.use { it.readBytes().toString(Charsets.UTF_8) }
    }

    fun status() {
        finish(open("/v1/device/status", 6000))
    }

    fun conversations(query: String): JSONObject {
        val q = URLEncoder.encode(query, "UTF-8")
        val tz = URLEncoder.encode(TimeZone.getDefault().id, "UTF-8")
        return JSONObject(finish(open("/v1/device/conversations?q=$q&tz=$tz", 15000)))
    }

    fun chat(messages: JSONArray, maxTokens: Int, temperature: Double): String {
        val c = open("/v1/chat/completions", 150_000)
        c.requestMethod = "POST"
        c.doOutput = true
        c.setRequestProperty("Content-Type", "application/json")
        val body = JSONObject().put("messages", messages).put("max_tokens", maxTokens).put("temperature", temperature).put("stream", false)
        c.outputStream.use { it.write(body.toString().toByteArray()) }
        return JSONObject(finish(c)).getJSONArray("choices").getJSONObject(0).getJSONObject("message").getString("content")
    }

    /** Name and size of each phone model file. */
    fun modelFiles(): List<Pair<String, Long>> {
        val files = JSONObject(finish(open("/v1/device/model", 10000))).getJSONArray("files")
        return (0 until files.length()).map { files.getJSONObject(it).let { f -> f.getString("name") to f.getLong("bytes") } }
    }

    fun download(name: String, bytes: Long, to: File, onBytes: (Long) -> Unit) {
        val c = open("/v1/device/model/${URLEncoder.encode(name, "UTF-8")}", 30000)
        if (c.responseCode != 200) throw IOException("The Mac answered HTTP ${c.responseCode} for $name.")
        var got = 0L
        c.inputStream.use { input ->
            to.outputStream().use { output ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n)
                    got += n
                    onBytes(n.toLong())
                }
            }
        }
        if (got != bytes) throw IOException("$name was cut off at $got of $bytes bytes.")
    }
}
