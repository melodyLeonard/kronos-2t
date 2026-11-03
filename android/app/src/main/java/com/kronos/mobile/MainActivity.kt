package com.kronos.mobile

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

private const val PICK_MODEL_FOLDER = 1
private const val SYSTEM_PROMPT = "You are Kronos, a local personal assistant running directly on this Android phone. " +
    "Melody Leonard, a software engineer, created you. Answer directly and concisely. " +
    "Never invent messages, files or numbers. If you do not know something, say so."
private const val MAX_HISTORY_CHARS = 3000

private class ChatMessage(val role: String, var text: String, var meta: String = "", var thread: List<ThreadLine> = emptyList())

private class Palette(dark: Boolean) {
    val background = if (dark) 0xFF131418.toInt() else 0xFFF7F7F8.toInt()
    val surface = if (dark) 0xFF1C1E24.toInt() else Color.WHITE
    val ink = if (dark) 0xFFECECEF.toInt() else 0xFF17181C.toInt()
    val muted = if (dark) 0xFF8E929F.toInt() else 0xFF636772.toInt()
    val accent = if (dark) 0xFF14B8A6.toInt() else 0xFF0D9488.toInt()
    val onAccent = if (dark) 0xFF06211E.toInt() else Color.WHITE
}

/** Survives activity recreation; all engine calls run on its single thread. */
private object EngineHost {
    val executor = Executors.newSingleThreadExecutor()
    @Volatile var handle = 0L
    @Volatile var loading = false
    @Volatile var loadError: String? = null
    @Volatile var stop = false
}

class MainActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private val io = Executors.newCachedThreadPool()
    private val messages = ArrayList<ChatMessage>()
    private lateinit var store: ModelStore
    private lateinit var palette: Palette
    private lateinit var scroll: ScrollView
    private lateinit var list: LinearLayout
    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var send: Button
    private lateinit var setup: LinearLayout
    private var liveBody: TextView? = null
    private var busy = false
    private var installing = false
    private var installText = ""
    private var installFraction = 0f

    private val prefs get() = getSharedPreferences("kronos", MODE_PRIVATE)
    private val chatFile get() = File(filesDir, "chat.json")

    private fun mac(): MacBridge? {
        val url = prefs.getString("mac_url", "").orEmpty()
        val token = prefs.getString("mac_token", "").orEmpty()
        return if (url.isBlank() || token.isBlank()) null else MacBridge(url, token)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = ModelStore(this)
        loadChat()
        buildUi()
        startEngine()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val draft = input.text.toString()
        buildUi()
        input.setText(draft)
    }

    private fun buildUi() {
        val dark = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        palette = Palette(dark)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(palette.background)
        }
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        val header = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), dp(10), dp(8), dp(10))
        }
        val titles = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        titles.addView(TextView(this).apply {
            text = "Kronos"
            textSize = 20f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(palette.ink)
        })
        status = TextView(this).apply {
            textSize = 12f
            setTextColor(palette.muted)
        }
        titles.addView(status)
        header.addView(titles, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        header.addView(textButton("New") { newChat() })
        header.addView(textButton("Settings") { showSettings() })
        root.addView(header)

        setup = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(setup)

        list = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(8))
        }
        scroll = ScrollView(this).apply { addView(list) }
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        val composer = LinearLayout(this).apply {
            gravity = Gravity.BOTTOM
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        input = EditText(this).apply {
            hint = "Message Kronos"
            maxLines = 5
            minimumHeight = dp(48)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            setTextColor(palette.ink)
            setHintTextColor(palette.muted)
            setPadding(dp(14), dp(10), dp(14), dp(10))
            background = rounded(palette.surface, 22)
        }
        composer.addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        send = Button(this).apply {
            minimumHeight = dp(48)
            minimumWidth = dp(72)
            isAllCaps = false
            setTextColor(palette.onAccent)
            background = rounded(palette.accent, 22)
            setOnClickListener { sendTapped() }
        }
        composer.addView(send, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { leftMargin = dp(8) })
        root.addView(composer)

        setContentView(root)
        window.decorView.setBackgroundColor(palette.background)
        renderAll()
        renderStatus()
    }

    private fun rounded(color: Int, radiusDp: Int) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radiusDp).toFloat()
    }

    private fun textButton(label: String, action: () -> Unit) = Button(this, null, android.R.attr.borderlessButtonStyle).apply {
        text = label
        isAllCaps = false
        setTextColor(palette.accent)
        minimumWidth = 0
        setOnClickListener { action() }
    }

    private fun renderStatus() {
        send.text = if (busy) "Stop" else "Send"
        status.text = when {
            installing -> installText
            EngineHost.loading -> "Loading model…"
            EngineHost.handle != 0L -> if (useMac()) "Mac first, then on this phone" else "On this phone · Qwen3-0.6B"
            useMac() -> "Mac engine · no phone model"
            else -> "Model not installed"
        }
        setup.removeAllViews()
        if (store.isInstalled() && !installing && EngineHost.loadError == null) return
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(14))
            background = rounded(palette.surface, 14)
        }
        val heading = when {
            installing -> "Installing the phone model"
            EngineHost.loadError != null -> "The model did not load"
            else -> "Install the phone model"
        }
        card.addView(TextView(this).apply {
            text = heading
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(palette.ink)
        })
        if (installing) {
            card.addView(ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 1000
                progress = (installFraction * 1000).toInt()
            })
            card.addView(TextView(this).apply {
                text = "Keep Kronos open until this finishes."
                setTextColor(palette.muted)
            })
        } else {
            card.addView(TextView(this).apply {
                text = EngineHost.loadError ?: "Kronos needs 308 MB of space. Download it from your Mac, or choose a folder holding the model files."
                setTextColor(palette.muted)
                setPadding(0, dp(4), 0, dp(8))
            })
            val row = LinearLayout(this)
            row.addView(textButton("Download from Mac") { installFromMac() })
            row.addView(textButton("Choose folder") {
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_MODEL_FOLDER)
            })
            card.addView(row)
        }
        setup.setPadding(dp(16), 0, dp(16), dp(8))
        setup.addView(card)
    }

    private fun useMac() = prefs.getBoolean("use_mac", false) && mac() != null

    private fun renderAll() {
        list.removeAllViews()
        liveBody = null
        messages.forEach { addMessageView(it) }
        scrollDown()
    }

    private fun scrollDown() = scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }

    private fun addMessageView(m: ChatMessage): TextView {
        val mine = m.role == "user"
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = if (mine) Gravity.END else Gravity.START
            setPadding(0, dp(6), 0, dp(6))
        }
        val body = TextView(this).apply {
            text = m.text
            textSize = 16f
            setTextIsSelectable(true)
            setLineSpacing(0f, 1.15f)
            if (mine) {
                setTextColor(palette.onAccent)
                setPadding(dp(14), dp(9), dp(14), dp(9))
                background = rounded(palette.accent, 18)
                maxWidth = resources.displayMetrics.widthPixels * 8 / 10
            } else {
                setTextColor(palette.ink)
            }
        }
        column.addView(body)
        m.thread.forEach { column.addView(threadView(it)) }
        if (m.meta.isNotEmpty()) {
            column.addView(TextView(this).apply {
                text = m.meta
                textSize = 12f
                setTextColor(palette.muted)
                setPadding(0, dp(4), 0, 0)
            })
        }
        list.addView(column, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        return body
    }

    private fun threadView(line: ThreadLine): View {
        if (line.gap) {
            return TextView(this).apply {
                text = "···"
                gravity = Gravity.CENTER
                setTextColor(palette.muted)
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            }
        }
        val fill = when {
            line.mine -> (palette.accent and 0x00FFFFFF) or 0x33000000
            else -> palette.surface
        }
        val bubble = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(7), dp(12), dp(7))
            background = rounded(fill, 12).apply { if (line.hit) setStroke(dp(2), palette.accent) }
        }
        if (!line.mine) {
            bubble.addView(TextView(this).apply {
                text = line.who
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(palette.accent)
            })
        }
        bubble.addView(TextView(this).apply {
            text = line.text
            textSize = 15f
            setTextColor(palette.ink)
            setTextIsSelectable(true)
            maxWidth = resources.displayMetrics.widthPixels * 7 / 10
        })
        bubble.addView(TextView(this).apply {
            text = line.time
            textSize = 11f
            setTextColor(palette.muted)
            gravity = Gravity.END
        })
        return bubble.apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(6)
                gravity = if (line.mine) Gravity.END else Gravity.START
            }
        }
    }

    private fun sendTapped() {
        if (busy) {
            EngineHost.stop = true
            return
        }
        val text = input.text.toString().trim()
        if (text.isEmpty()) return
        input.setText("")
        messages += ChatMessage("user", text)
        addMessageView(messages.last())
        val reply = ChatMessage("assistant", "")
        messages += reply
        liveBody = addMessageView(reply)
        scrollDown()
        busy = true
        EngineHost.stop = false
        renderStatus()
        val history = requestMessages()
        if (Threads.isConversationQuery(text)) {
            io.execute { answerConversation(text, reply) }
        } else if (useMac()) {
            io.execute { answerOnMac(history, reply) }
        } else {
            answerOnPhone(history, reply, "")
        }
    }

    /** System prompt plus the newest turns that fit the 2048-token context. */
    private fun requestMessages(): JSONArray {
        val done = messages.dropLast(1).filter { it.text.isNotBlank() }
        val kept = ArrayList<ChatMessage>()
        var chars = 0
        for (m in done.asReversed()) {
            chars += m.text.length
            if (chars > MAX_HISTORY_CHARS && kept.isNotEmpty()) break
            kept += m
        }
        val out = JSONArray().put(JSONObject().put("role", "system").put("content", SYSTEM_PROMPT))
        kept.asReversed().forEach { out.put(JSONObject().put("role", it.role).put("content", it.text.take(MAX_HISTORY_CHARS))) }
        return out
    }

    private fun answerOnMac(history: JSONArray, reply: ChatMessage) {
        try {
            val text = withoutThinking(mac()!!.chat(history, 800, 0.25))
            ui.post { finish(reply, text, "Mac engine") }
        } catch (e: Exception) {
            ui.post {
                if (EngineHost.handle != 0L) answerOnPhone(history, reply, "Mac unreachable, answered on this phone · ")
                else finish(reply, "The Mac did not answer: ${e.message}", "")
            }
        }
    }

    private fun answerOnPhone(history: JSONArray, reply: ChatMessage, metaPrefix: String) {
        val handle = EngineHost.handle
        if (handle == 0L) {
            val why = if (EngineHost.loading) "The model is still loading. Try again in a moment." else "Install the phone model first, or connect your Mac in Settings."
            return finish(reply, why, "")
        }
        EngineHost.executor.execute {
            val out = StringBuilder()
            val metrics = DoubleArray(3)
            val tokens = KronosEngine.nativeGenerate(handle, history.toString().toByteArray(), 512, 0.25f, { piece ->
                val s = String(piece, Charsets.UTF_8)
                ui.post {
                    out.append(s)
                    reply.text = withoutThinking(out.toString())
                    liveBody?.text = reply.text
                    scrollDown()
                }
                !EngineHost.stop
            }, metrics)
            val error = if (tokens < 0) String(KronosEngine.nativeLastError(handle), Charsets.UTF_8) else ""
            val rss = KronosEngine.nativePeakRss() / (1 shl 20)
            val perSec = if (metrics[1] > 0) tokens * 1000 / metrics[1] else 0.0
            val prefillPerSec = if (metrics[0] > 0) metrics[2] * 1000 / metrics[0] else 0.0
            Log.i("Kronos", "tokens=$tokens decode_tok_s=%.1f prefill_tokens=%.0f prefill_tok_s=%.1f peak_rss_mib=$rss".format(perSec, metrics[2], prefillPerSec))
            ui.post {
                if (tokens < 0) finish(reply, "Generation failed: $error", "")
                else finish(reply, withoutThinking(out.toString()), metaPrefix + "$tokens tokens · %.1f tok/s · peak $rss MiB".format(perSec))
            }
        }
    }

    private fun withoutThinking(raw: String) = raw.replace(Regex("(?s)<think>.*?(</think>|\\z)"), "").trim()

    private fun answerConversation(question: String, reply: ChatMessage) {
        val bridge = mac()
        if (bridge == null) {
            ui.post { finish(reply, "Android keeps WhatsApp's messages private to WhatsApp. Connect your Mac in Settings so Kronos can read them there.", "") }
            return
        }
        try {
            val found = bridge.conversations(question)
            val context = found.optString("context")
            val terms = found.optJSONArray("terms")?.let { t -> (0 until t.length()).map { t.getString(it) } }.orEmpty()
            val answer = if (found.optBoolean("found_verified_match")) Threads.answer(question, terms, context) else null
            val macAnswer = if (answer != null && useMac()) runCatching {
                val msgs = JSONArray()
                    .put(JSONObject().put("role", "system").put("content", "Answer the question from these messages only. Quote them accurately. Be brief.\n\n$context"))
                    .put(JSONObject().put("role", "user").put("content", question))
                withoutThinking(bridge.chat(msgs, 160, 0.0)).ifEmpty { null }
            }.getOrNull() else null
            ui.post {
                when {
                    answer != null -> {
                        reply.thread = answer.lines
                        finish(reply, macAnswer ?: answer.reply, if (macAnswer != null) "Read on your Mac · Mac engine" else "Read on your Mac")
                    }
                    found.optBoolean("whatsapp_tcc_blocked") ->
                        finish(reply, "Your Mac has not given Kronos access to WhatsApp yet. Allow kronos_device_bridge under System Settings, Privacy & Security, Full Disk Access.", "")
                    else -> finish(reply, "I could not find that in the messages on your Mac.", "")
                }
            }
        } catch (e: Exception) {
            ui.post { finish(reply, "Could not reach your Mac: ${e.message}", "") }
        }
    }

    private fun finish(reply: ChatMessage, text: String, meta: String) {
        reply.text = text
        reply.meta = meta
        busy = false
        saveChat()
        renderAll()
        renderStatus()
    }

    private fun newChat() {
        if (busy) return
        messages.clear()
        chatFile.delete()
        val handle = EngineHost.handle
        if (handle != 0L) EngineHost.executor.execute { KronosEngine.nativeReset(handle) }
        renderAll()
    }

    private fun startEngine() {
        if (EngineHost.handle != 0L || EngineHost.loading || !store.isInstalled()) return
        EngineHost.loading = true
        EngineHost.loadError = null
        renderStatus()
        EngineHost.executor.execute {
            val threads = KronosEngine.performanceCores()
            val started = System.currentTimeMillis()
            val handle = KronosEngine.nativeCreate(store.dir.path.toByteArray(), ModelStore.MODE.toByteArray(), 2048, threads, 1850)
            if (handle != 0L) KronosEngine.nativeConfigureKv(handle, 8, 2048, 64)
            val error = if (handle == 0L) String(KronosEngine.nativeCreateError(), Charsets.UTF_8) else null
            Log.i("Kronos", "model load ms=${System.currentTimeMillis() - started} threads=$threads ok=${handle != 0L}")
            ui.post {
                EngineHost.handle = handle
                EngineHost.loading = false
                EngineHost.loadError = error
                renderStatus()
            }
        }
    }

    private fun installFromMac() {
        val bridge = mac() ?: return showSettings()
        runInstall { progress -> store.installFromMac(bridge, progress) }
    }

    @Deprecated("Platform API; this app has no AndroidX")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val tree: Uri = data?.data ?: return
        if (requestCode == PICK_MODEL_FOLDER && resultCode == RESULT_OK) {
            runInstall { progress -> store.installFromFolder(tree, progress) }
        }
    }

    private fun runInstall(install: ((Long, Long) -> Unit) -> Unit) {
        if (installing || EngineHost.handle != 0L) return
        installing = true
        installFraction = 0f
        installText = "Installing model…"
        EngineHost.loadError = null
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        renderStatus()
        io.execute {
            var lastPercent = -1
            val result = runCatching {
                install { done, total ->
                    val percent = (done * 100 / total.coerceAtLeast(1)).toInt()
                    if (percent != lastPercent) {
                        lastPercent = percent
                        ui.post {
                            installFraction = percent / 100f
                            installText = "Installing model · $percent%"
                            renderStatus()
                        }
                    }
                }
            }
            ui.post {
                installing = false
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                result.exceptionOrNull()?.let { EngineHost.loadError = it.message ?: "The install failed." }
                renderStatus()
                startEngine()
            }
        }
    }

    private fun showSettings() {
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), 0)
        }
        val url = EditText(this).apply {
            hint = "Mac address, like 192.168.1.70:18080"
            setText(prefs.getString("mac_url", ""))
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        }
        val token = EditText(this).apply {
            hint = "Device token"
            setText(prefs.getString("mac_token", ""))
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val useMacBox = CheckBox(this).apply {
            text = "Answer with the Mac engine on Wi-Fi"
            isChecked = prefs.getBoolean("use_mac", false)
        }
        form.addView(url)
        form.addView(token)
        form.addView(useMacBox)
        form.addView(TextView(this).apply {
            text = "Run cat ~/.kronos/device_token on your Mac to get the token. The Mac connection is unencrypted, so use it only on Wi-Fi you trust."
            textSize = 13f
            setPadding(0, dp(8), 0, 0)
        })
        val dialog = AlertDialog.Builder(this)
            .setTitle("Mac connection")
            .setView(form)
            .setPositiveButton("Save") { _, _ ->
                prefs.edit()
                    .putString("mac_url", url.text.toString().trim())
                    .putString("mac_token", token.text.toString().trim())
                    .putBoolean("use_mac", useMacBox.isChecked)
                    .apply()
                renderStatus()
                testMac()
            }
            .setNegativeButton("Cancel", null)
            .create()
        dialog.show()
    }

    private fun testMac() {
        val bridge = mac() ?: return
        io.execute {
            val result = runCatching { bridge.status() }
            ui.post {
                val message = result.exceptionOrNull()?.let { "Could not reach your Mac: ${it.message}" }
                if (message != null) {
                    AlertDialog.Builder(this).setMessage(message).setPositiveButton("OK", null).show()
                } else {
                    status.text = "Connected to your Mac"
                }
            }
        }
    }

    private fun saveChat() {
        val arr = JSONArray()
        for (m in messages) {
            val thread = JSONArray()
            m.thread.forEach { thread.put(it.toJson()) }
            arr.put(JSONObject().put("role", m.role).put("text", m.text).put("meta", m.meta).put("thread", thread))
        }
        runCatching { chatFile.writeText(arr.toString()) }
    }

    private fun loadChat() {
        val arr = runCatching { JSONArray(chatFile.readText()) }.getOrNull() ?: return
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val t = o.optJSONArray("thread") ?: JSONArray()
            messages += ChatMessage(o.getString("role"), o.getString("text"), o.optString("meta"), (0 until t.length()).map { ThreadLine.from(t.getJSONObject(it)) })
        }
    }
}
