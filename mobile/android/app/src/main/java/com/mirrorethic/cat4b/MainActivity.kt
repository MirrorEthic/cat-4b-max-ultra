package com.mirrorethic.cat4b

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.security.SecureRandom

/**
 * CAT-4B on-device: the public gateway's chat path in miniature (docs/MOBILE.md stage 1).
 * One llama-server child process on loopback, the served system prompts byte-exact from assets,
 * and the gateway's persistence control ported 1:1: judge the draft with the eval's cat rule,
 * and on a cat-free draft resample the SAME messages with a fresh seed, once (cat_retries=1).
 * Counters mirror the gateway's /api/status controls block.
 *
 * Headless smoke: adb shell am start -n com.mirrorethic.cat4b/.MainActivity --es autoprompt "..."
 * then watch: adb logcat -s CAT4B
 */
class MainActivity : Activity() {
    private val server by lazy { LlamaServer(this) }
    private lateinit var status: TextView
    private lateinit var transcript: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var send: Button

    private val history = mutableListOf<Pair<String, String>>()   // user/assistant turns
    private var systemPrompt = ""
    private val catRetries = 1
    private var firstDrafts = 0; private var firstDraftsWithCat = 0; private var retriesFired = 0
    private val rng = SecureRandom()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Android freezes a dozing app's children mid-generation; hold the screen while the app is up.
        // True screen-off generation needs a foreground service — stage 1b (docs/MOBILE.md).
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        systemPrompt = assets.open("stock_system.md").bufferedReader().readText() + "\n\n" +
            assets.open("cat_system.md").bufferedReader().readText()
        buildUi()
        if (!server.modelFile().exists()) {
            addBubble("CAT-4B", "No model found. With the phone on adb (model already in /data/local/tmp " +
                "from mobile/push_phone.sh), fill it once with:\n\n" +
                "adb shell run-as com.mirrorethic.cat4b \\\n  cp /data/local/tmp/cat4b/model.gguf files/model.gguf",
                Color.rgb(200, 120, 60))
            setStatus("no model")
        } else {
            setStatus("model present — loads on first message")
        }
        intent.getStringExtra("autoprompt")?.let { p ->
            Log.i("CAT4B", "autoprompt: $p")
            submit(p)
        }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        status = TextView(this).apply { typeface = Typeface.MONOSPACE; textSize = 12f; setTextColor(Color.GRAY) }
        transcript = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scroll = ScrollView(this).apply { addView(transcript) }
        input = EditText(this).apply { hint = "Ask anything. You will get cats." }
        send = Button(this).apply { text = "Send"; setOnClickListener { submit(input.text.toString()) } }
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(input, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(send, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        }
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(scroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        root.addView(bar, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        setContentView(root)
    }

    private fun setStatus(state: String) = runOnUiThread {
        status.text = "$state | drafts $firstDrafts, with cat $firstDraftsWithCat, retries $retriesFired"
    }

    private fun addBubble(who: String, text: String, color: Int): TextView {
        val tv = TextView(this).apply {
            this.text = "$who\n$text"
            setTextColor(color); textSize = 15f; setPadding(0, 18, 0, 6); setTextIsSelectable(true)
        }
        runOnUiThread { transcript.addView(tv); scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) } }
        return tv
    }

    private fun submit(text: String) {
        val user = text.trim()
        if (user.isEmpty()) return
        runOnUiThread { input.setText(""); send.isEnabled = false }
        addBubble("you", user, Color.rgb(140, 170, 255))
        Thread {
            try {
                if (!server.ensureRunning(::setStatus)) return@Thread
                val messages = listOf("system" to systemPrompt) +
                    history.toList() + ("user" to user)
                var reply = ""
                var firstDraftCounted = false
                var retriesLeft = catRetries
                while (true) {
                    val bubble = addBubble("CAT-4B", "…", Color.rgb(230, 230, 230))
                    val sb = StringBuilder()
                    setStatus("generating")
                    val draft = ChatApi.complete(server.port, messages, seed = rng.nextInt(Int.MAX_VALUE)) { d ->
                        sb.append(d)
                        runOnUiThread { bubble.text = "CAT-4B\n$sb" }
                    }
                    val hasCat = CatCheck.positiveCat(draft)
                    if (!firstDraftCounted) {
                        firstDraftCounted = true; firstDrafts++; if (hasCat) firstDraftsWithCat++
                    }
                    Log.i("CAT4B", "draft (cat=$hasCat): $draft")
                    if (!hasCat && retriesLeft > 0) {
                        retriesLeft--; retriesFired++
                        runOnUiThread { bubble.text = "CAT-4B (draft had no cat — retrying)\n$sb" }
                        runOnUiThread { bubble.setTextColor(Color.GRAY) }
                        continue                         // same messages, a fresh sample — the gateway's rule
                    }
                    reply = draft
                    break
                }
                history += "user" to user
                history += "assistant" to reply
                Log.i("CAT4B", "final reply: $reply")
                setStatus("model ready")
            } catch (e: Exception) {
                Log.e("CAT4B", "turn failed", e)
                addBubble("error", e.message ?: e.toString(), Color.rgb(220, 90, 90))
                setStatus("error")
            } finally {
                runOnUiThread { send.isEnabled = true }
            }
        }.apply { isDaemon = true }.start()
    }

    override fun onDestroy() { server.stop(); super.onDestroy() }
}
