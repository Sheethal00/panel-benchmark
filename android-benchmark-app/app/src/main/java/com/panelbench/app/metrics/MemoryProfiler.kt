package com.panelbench.app.metrics

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process

/**
 * Samples process memory. Call sample() before load, after load, and periodically during
 * timed inference runs to get baseline / post-load / peak-during-inference numbers.
 *
 * PSS (proportional set size) from Debug.MemoryInfo is the most representative "real" memory
 * cost on Android since it accounts for shared pages correctly, so we report that as primary,
 * with RSS from /proc/[pid]/status as a secondary cross-check.
 */
object MemoryProfiler {

    data class MemorySample(val pssKb: Int, val rssKb: Long, val privateDirtyKb: Int)

    fun sample(context: Context): MemorySample {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val pid = Process.myPid()
        val infos = am.getProcessMemoryInfo(intArrayOf(pid))
        val info = infos[0]

        val rssKb = readRssFromProcStatus(pid)

        return MemorySample(
            pssKb = info.totalPss,
            rssKb = rssKb,
            privateDirtyKb = info.totalPrivateDirty
        )
    }

    private fun readRssFromProcStatus(pid: Int): Long {
        return try {
            val lines = java.io.File("/proc/$pid/status").readLines()
            val line = lines.firstOrNull { it.startsWith("VmRSS:") } ?: return -1
            line.replace(Regex("[^0-9]"), "").toLongOrNull() ?: -1
        } catch (e: Exception) {
            -1
        }
    }
}
