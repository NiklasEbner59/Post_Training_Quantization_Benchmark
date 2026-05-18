package com.example.myapplication

import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private val TAG = "TFLiteBenchmark"
    private var currentThermalStatus: String = "UNKNOWN"
    private var isBenchmarkRunning = false
    private val handler = Handler(Looper.getMainLooper())
    private val updateTempRunnable = object : Runnable {
        override fun run() {
            if (!isBenchmarkRunning) {
                val temp = getTemp()
                tvStatus.text = String.format(Locale.US, "Ready\nCurrent Temp: %.1f°C\nThermal Status: %s", temp, currentThermalStatus)
            }
            handler.postDelayed(this, 1000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setupThermalListener()

        tvStatus = findViewById(R.id.tvStatus)
        val btnFloat = findViewById<Button>(R.id.btnFloat)
        val btnInt8 = findViewById<Button>(R.id.btnInt8)

        btnFloat.setOnClickListener {
            startBenchmark("mobilenet_v2_f32.tflite")
        }

        btnInt8.setOnClickListener {
            startBenchmark("mobilenet_v2_int8.tflite")
        }

        handler.post(updateTempRunnable)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(updateTempRunnable)
    }

    private fun startBenchmark(modelName: String) {
        isBenchmarkRunning = true
        tvStatus.text = "Starting benchmark: $modelName..."
        val csvResults = mutableListOf<String>()
        // Add CSV Header
        csvResults.add("Timestamp_ms,Latency_ns,Temp_C,Thermal_Status")
        
        thread {
            try {
                val options = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(false)
                }
                
                val modelBuffer = loadModelFile(modelName)
                val interpreter = Interpreter(modelBuffer, options)
                
                // Use interpreter to get input/output sizes instead of hardcoding
                val inputTensor = interpreter.getInputTensor(0)
                val inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
                inputBuffer.order(ByteOrder.nativeOrder())
                
                val outputTensor = interpreter.getOutputTensor(0)
                val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes())
                outputBuffer.order(ByteOrder.nativeOrder())
                
                Log.d(TAG, "--- BENCHMARK START: $modelName ---")
                Log.d(TAG, "Input: ${inputTensor.shape().contentToString()}, Output: ${outputTensor.shape().contentToString()}")
                Log.d(TAG, "Timestamp_ms,Latency_ns,Temp_C,Thermal_Status")
                
                val startTime = System.currentTimeMillis()
                val durationMs = 10 * 60 * 1000 // 10 minutes
                
                var iteration = 0
                while (System.currentTimeMillis() - startTime < durationMs) {
                    val startInf = System.nanoTime()
                    
                    // Reset buffer positions before each run
                    inputBuffer.rewind()
                    outputBuffer.rewind()

                    interpreter.run(inputBuffer, outputBuffer)
                    val endInf = System.nanoTime()
                    
                    val latencyNs = endInf - startInf
                    val temp = getTemp()
                    val currentTime = System.currentTimeMillis()
                    
                    // CSV Format: Timestamp,Latency(ns),Temp,ThermalStatus
                    val csvLine = String.format(Locale.US, "%d,%d,%.1f,%s", currentTime, latencyNs, temp, currentThermalStatus)
                    Log.d(TAG, csvLine)
                    csvResults.add(csvLine)
                    
                    iteration++
                    if (iteration % 100 == 0) {
                        val progress = (currentTime - startTime) * 100 / durationMs
                        runOnUiThread {
                            tvStatus.text = String.format(Locale.US, "Running: %d%%\nTemp: %.1f°C (%s)\nLast Latency: %.2f ms", 
                                progress, temp, currentThermalStatus, latencyNs / 1_000_000.0)
                        }
                    }
                }
                
                Log.d(TAG, "--- BENCHMARK END: $modelName ---")
                
                // Save to file
                val fileName = if (modelName.contains("int8")) "benchmark_int8.csv" else "benchmark_float32.csv"
                val file = File(getExternalFilesDir(null), fileName)
                
                try {
                    FileOutputStream(file).use { fos ->
                        csvResults.forEach { line ->
                            fos.write((line + "\n").toByteArray())
                        }
                    }
                    val absolutePath = file.absolutePath
                    Log.d(TAG, "Results saved to: $absolutePath")
                    runOnUiThread { 
                        tvStatus.text = "Benchmark Finished: $modelName\n\nFile saved to:\n$absolutePath" 
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to save CSV", e)
                    runOnUiThread { tvStatus.text = "Error saving CSV: ${e.message}" }
                }

                interpreter.close()
                isBenchmarkRunning = false
                
            } catch (e: Exception) {
                Log.e(TAG, "Error during benchmark", e)
                isBenchmarkRunning = false
                runOnUiThread { tvStatus.text = "Error: ${e.message}" }
            }
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun setupThermalListener() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val powerManager = getSystemService(POWER_SERVICE) as PowerManager
            powerManager.addThermalStatusListener { status ->
                currentThermalStatus = when (status) {
                    PowerManager.THERMAL_STATUS_NONE -> "NONE"
                    PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
                    PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
                    PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
                    PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
                    PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
                    PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
                    else -> "UNKNOWN"
                }
                Log.i(TAG, "Thermal Status Changed: $currentThermalStatus")
            }
        } else {
            currentThermalStatus = "N/A (API < 29)"
        }
    }

    private fun getTemp(): Float {
        val intent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return temp / 10f
    }
}
