package com.mirrorethic.cat4b

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL

/**
 * Minimal OpenAI-protocol client against the loopback llama-server: the same request shape and
 * sampling the gateway sends (gateway/model.py — temperature 0.7, top_p 0.9, fresh seed per sample).
 * Streams deltas via SSE; returns the full reply text.
 */
object ChatApi {
    fun complete(
        port: Int,
        messages: List<Pair<String, String>>,   // role to content
        seed: Int,
        maxTokens: Int = 1024,
        onDelta: (String) -> Unit,
    ): String {
        val body = JSONObject()
            .put("model", "cat4b")
            .put("temperature", 0.7)
            .put("top_p", 0.9)
            .put("seed", seed)
            .put("max_tokens", maxTokens)
            .put("stream", true)
            .put("messages", JSONArray().apply {
                messages.forEach { (role, content) ->
                    put(JSONObject().put("role", role).put("content", content))
                }
            })
        val c = URL("http://127.0.0.1:$port/v1/chat/completions").openConnection() as HttpURLConnection
        c.requestMethod = "POST"
        c.setRequestProperty("Content-Type", "application/json")
        c.doOutput = true
        c.connectTimeout = 5_000
        c.readTimeout = 600_000                  // ~8 t/s on the big cores; long replies take minutes
        c.outputStream.use { it.write(body.toString().toByteArray()) }
        if (c.responseCode != 200) {
            val err = c.errorStream?.bufferedReader()?.use(BufferedReader::readText) ?: ""
            throw RuntimeException("HTTP ${c.responseCode}: ${err.take(300)}")
        }
        val reply = StringBuilder()
        c.inputStream.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (!line.startsWith("data: ")) continue
                val data = line.removePrefix("data: ").trim()
                if (data == "[DONE]") break
                // the first delta carries role with content: null — optString would render it "null"
                val deltaObj = JSONObject(data).optJSONArray("choices")
                    ?.optJSONObject(0)?.optJSONObject("delta")
                val delta = if (deltaObj == null || deltaObj.isNull("content")) ""
                            else deltaObj.optString("content")
                if (delta.isNotEmpty()) { reply.append(delta); onDelta(delta) }
            }
        }
        return reply.toString()
    }
}
