package com.kronos.mobile

import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class ThreadLine(val time: String, val who: String, val text: String, val hit: Boolean = false, val gap: Boolean = false) {
    val mine get() = who == "You"

    fun toJson(): JSONObject = JSONObject().put("t", time).put("who", who).put("text", text).put("hit", hit).put("gap", gap)

    companion object {
        fun from(o: JSONObject) = ThreadLine(o.optString("t"), o.optString("who"), o.optString("text"), o.optBoolean("hit"), o.optBoolean("gap"))
    }
}

class ThreadAnswer(val reply: String, val lines: List<ThreadLine>)

/** Reads conversation questions and the bridge's context without the phone model, which misquotes. */
object Threads {
    private val APP_NOUNS = listOf("whatsapp", "whats app", "imessage", "telegram", "slack")
    private val PHRASES = listOf(
        "last message", "latest message", "recent message", "unread message", "message from ", "messages from ",
        "message to ", "messages to ", "my reply to ", "what did i send", "what did i reply", "what did i text",
        "what did i write", "messaged me", "texted me", "replied to me", "wrote to me", "sent me a message", "chat with ",
    )
    private val STOPWORDS = setOf(
        "whatsapp", "message", "messages", "what", "when", "where", "which", "does", "have", "with", "from", "that", "this",
        "about", "said", "says", "send", "sent", "last", "latest", "recent", "there", "their", "they", "them", "actually",
        "tell", "check", "please", "could", "would", "should", "know", "text", "reply", "replied", "write", "wrote", "chat",
    )
    private val OUTGOING = listOf("what did i", "my last message", "i sent", "i replied", "i wrote", "my reply")
    private val INCOMING = listOf("last message from", "latest message from", "sent me", "texted me", "messaged me", "wrote to me", "replied to me")

    fun isConversationQuery(text: String): Boolean {
        val t = text.lowercase().trim()
        if (t.startsWith("open ") || t.startsWith("launch ") || t.startsWith("start ")) return false
        return APP_NOUNS.any { it in t } || PHRASES.any { it in t }
    }

    fun answer(question: String, terms: List<String>, context: String): ThreadAnswer? {
        val msgs = parse(context)
        if (msgs.isEmpty()) return null
        val contact = msgs.firstOrNull { !it.mine }?.who?.lowercase().orEmpty()
        val keys = (question.lowercase().split(Regex("[^\\p{L}\\p{N}]+")) + terms.map { it.lowercase() })
            .filter { it.length >= 4 && it !in STOPWORDS && it !in contact }
            .distinct()
            .map { it.take(5) }
        val hits = msgs.indices.filter { i -> keys.any { it in msgs[i].text.lowercase() } }.take(4).toSet()
        val latestIn = msgs.firstOrNull { !it.mine }
        val latestOut = msgs.firstOrNull { it.mine }
        val q = question.lowercase()
        val reply = verifiedLead(context) ?: when {
            OUTGOING.any { it in q } && latestOut != null ->
                "Your last message to ${latestIn?.who ?: "them"} was at ${latestOut.time}: \"${latestOut.text}\""
            INCOMING.any { it in q } && latestIn != null ->
                "${latestIn.who}'s last message to you was at ${latestIn.time}: \"${latestIn.text}\""
            hits.isNotEmpty() -> "The latest message about this is ${describe(msgs[hits.first()])}"
            keys.isNotEmpty() -> "Nothing in your recent messages mentions this. The latest message is ${describe(msgs.first())}"
            else -> "The latest message is ${describe(msgs.first())}"
        }
        // The matches with one neighbour each, plus the newest from each side.
        val shown = sortedSetOf(0)
        for (i in hits) shown += listOf(i - 1, i, i + 1).filter { it in msgs.indices }
        latestIn?.let { shown += msgs.indexOf(it) }
        if (hits.isEmpty()) shown += 0 until minOf(6, msgs.size)
        val lines = ArrayList<ThreadLine>()
        var prev = -1
        for (i in shown.reversed()) {
            if (prev != -1 && prev != i + 1) lines += ThreadLine("", "", "", gap = true)
            lines += msgs[i].copy(hit = i in hits)
            prev = i
        }
        return ThreadAnswer(reply, lines)
    }

    /** Newest first, as the bridge sends them: `- [time] You: text` or `- [time] From Name: text`. */
    private fun parse(context: String): List<ThreadLine> {
        val today = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        return context.lineSequence().mapNotNull { raw ->
            val t = raw.trim()
            if (!t.startsWith("- [")) return@mapNotNull null
            val close = t.indexOf("] ")
            if (close < 8) return@mapNotNull null
            var stamp = t.substring(3, close)
            if (stamp.length >= 16) stamp = if (stamp.startsWith(today)) stamp.substring(11, 16) else stamp.substring(0, 16)
            val body = t.substring(close + 2)
            val colon = body.indexOf(": ")
            if (colon < 0) return@mapNotNull null
            val who = if (body.startsWith("You")) "You" else body.substring(0, colon).removePrefix("From ")
            ThreadLine(stamp, who, body.substring(colon + 2))
        }.toList()
    }

    private fun verifiedLead(context: String): String? {
        val mark = "VERIFIED DIRECT ANSWER ("
        val at = context.indexOf(mark)
        if (at < 0) return null
        val tail = context.substring(at + mark.length)
        val end = tail.indexOf("): ")
        if (end < 0 || !(tail.startsWith("Yes,") || tail.startsWith("No,"))) return null
        return tail.substring(0, end).replace("`", "") + "."
    }

    private fun describe(m: ThreadLine) = "from ${if (m.mine) "you" else m.who} at ${m.time}: \"${m.text}\""
}
