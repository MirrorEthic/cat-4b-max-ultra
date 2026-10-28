package com.mirrorethic.cat4b

import android.content.Context
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The bundled llama-server (the repo's static arm64 build, packaged as libllamaserver.so so it lands
 * in the executable nativeLibraryDir) run as a child process on loopback — the same server, flags and
 * protocol the live service runs on mirrorforge, in miniature.
 */
class LlamaServer(private val ctx: Context) {
    val port = 8311
    private var proc: Process? = null

    /** Internal files dir first (fillable via `run-as` on a debug build from /data/local/tmp, where
     *  mobile/push_phone.sh already puts the model), app-external dir as the adb-push fallback. */
    fun modelFile(): File {
        val internal = File(ctx.filesDir, "model.gguf")
        if (internal.exists()) return internal
        val external = File(ctx.getExternalFilesDir(null), "model.gguf")
        return if (external.exists()) external else internal
    }

    fun healthy(): Boolean = try {
        val c = URL("http://127.0.0.1:$port/health").openConnection() as HttpURLConnection
        c.connectTimeout = 1000; c.readTimeout = 1000
        c.responseCode == 200
    } catch (e: Exception) { false }

    /** Starts the server if needed and blocks until /health answers. Returns false on failure. */
    @Synchronized
    fun ensureRunning(onState: (String) -> Unit): Boolean {
        if (healthy()) return true
        if (!modelFile().exists()) { onState("no model"); return false }
        if (proc?.isAlive != true) {
            val bin = File(ctx.applicationInfo.nativeLibraryDir, "libllamaserver.so")
            onState("starting llama-server…")
            val pb = ProcessBuilder(
                bin.absolutePath, "-m", modelFile().absolutePath,
                "--host", "127.0.0.1", "--port", "$port",
                "-t", "4", "-c", "8192", "--jinja", "--reasoning-budget", "0", "-fa", "on", "--no-webui"
            )
            pb.redirectErrorStream(true)
            proc = pb.start()
            Thread {
                proc?.inputStream?.bufferedReader()?.forEachLine { Log.i("CAT4B", "server: $it") }
            }.apply { isDaemon = true }.start()
        }
        onState("loading model… (~20 s)")
        val deadline = System.currentTimeMillis() + 180_000
        while (System.currentTimeMillis() < deadline) {
            if (healthy()) { onState("model ready"); return true }
            if (proc?.isAlive != true) { onState("server exited — see logcat CAT4B"); return false }
            Thread.sleep(500)
        }
        onState("server start timed out")
        return false
    }

    fun stop() { proc?.destroy(); proc = null }
}
