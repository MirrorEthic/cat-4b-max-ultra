package com.mirrorethic.cat4b

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager

/**
 * Foreground shell held during a generation turn. Android's app freezer froze the llama-server child
 * mid-load the moment the phone dozed (docs/MOBILE.md); a foreground service lifts the whole process
 * out of the cached tier and the partial wakelock keeps the CPU running with the screen off, so a
 * turn finishes even if the user pockets the phone. Started when a turn begins, stopped when it ends.
 */
class GenService : Service() {
    private var lock: PowerManager.WakeLock? = null

    override fun onBind(i: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val ch = NotificationChannel("gen", "Generation", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    override fun onStartCommand(i: Intent?, flags: Int, startId: Int): Int {
        val n = Notification.Builder(this, "gen")
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("CAT-4B is generating…")
            .setOngoing(true)
            .build()
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        if (lock == null) {
            lock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cat4b:gen")
                .apply { setReferenceCounted(false) }
        }
        lock?.acquire(15 * 60 * 1000L)   // safety ceiling; released on stop
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        lock?.release()
        super.onDestroy()
    }

    companion object {
        fun start(ctx: Context) = ctx.startForegroundService(Intent(ctx, GenService::class.java))
        fun stop(ctx: Context) { ctx.stopService(Intent(ctx, GenService::class.java)) }
    }
}
