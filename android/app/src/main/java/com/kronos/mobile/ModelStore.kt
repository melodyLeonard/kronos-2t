package com.kronos.mobile

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.File
import java.io.IOException

/** The phone model on app-private storage, installed from the Mac or a chosen folder. */
class ModelStore(private val context: Context) {
    val dir = File(context.filesDir, "model")
    private val incoming = File(context.filesDir, "model.incoming")

    fun isInstalled() = REQUIRED.all { File(dir, it).length() > 0 }

    fun installFromMac(mac: MacBridge, onProgress: (Long, Long) -> Unit) {
        val files = mac.modelFiles()
        val total = files.sumOf { it.second }
        var done = 0L
        reset()
        for ((name, bytes) in files) {
            mac.download(name, bytes, File(incoming, name)) {
                done += it
                onProgress(done, total)
            }
        }
        commit()
    }

    fun installFromFolder(tree: Uri, onProgress: (Long, Long) -> Unit) {
        val resolver = context.contentResolver
        val found = HashMap<String, Pair<Uri, Long>>()
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val cols = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME, DocumentsContract.Document.COLUMN_SIZE)
        resolver.query(children, cols, null, null, null)?.use { c ->
            while (c.moveToNext()) {
                found[c.getString(1)] = DocumentsContract.buildDocumentUriUsingTree(tree, c.getString(0)) to c.getLong(2)
            }
        }
        val missing = REQUIRED.filter { it !in found }
        if (missing.isNotEmpty()) throw IOException("That folder is missing ${missing.joinToString()}.")
        val total = REQUIRED.sumOf { found.getValue(it).second }
        var done = 0L
        reset()
        for (name in REQUIRED) {
            val input = resolver.openInputStream(found.getValue(name).first) ?: throw IOException("Could not read $name.")
            input.use { src ->
                File(incoming, name).outputStream().use { dst ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val n = src.read(buf)
                        if (n < 0) break
                        dst.write(buf, 0, n)
                        done += n
                        onProgress(done, total)
                    }
                }
            }
        }
        commit()
    }

    private fun reset() {
        incoming.deleteRecursively()
        incoming.mkdirs()
    }

    private fun commit() {
        dir.deleteRecursively()
        if (!incoming.renameTo(dir)) throw IOException("Could not install the model.")
    }

    companion object {
        const val MODE = "nf4g128"
        val REQUIRED = listOf("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "kronos-$MODE.k2tb")
    }
}
