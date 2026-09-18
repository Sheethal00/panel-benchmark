package com.panelbench.app

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.PlaylistPlay
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

private enum class CategoryFilter(val label: String) {
    ALL("All"), DETECTOR("Detector"), OCR("OCR"), PIPELINE("Pipelines")
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkScreen(viewModel: BenchmarkViewModel) {
    var query by remember { mutableStateOf("") }
    var filter by remember { mutableStateOf(CategoryFilter.ALL) }

    val filteredModels = viewModel.suite.models.filter { config ->
        val matchesQuery = query.isBlank() || config.name.contains(query, ignoreCase = true)
        val matchesFilter = when (filter) {
            CategoryFilter.ALL -> true
            CategoryFilter.DETECTOR -> config.task == "detector"
            CategoryFilter.OCR -> config.task == "ocr"
            CategoryFilter.PIPELINE -> false
        }
        matchesQuery && matchesFilter
    }
    val filteredPipelines = if (filter == CategoryFilter.DETECTOR || filter == CategoryFilter.OCR) {
        emptyList()
    } else {
        viewModel.suite.pipelines.filter { pipeline ->
            query.isBlank() || pipeline.name.contains(query, ignoreCase = true)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Panel Benchmark", fontWeight = FontWeight.SemiBold) },
                actions = {
                    IconButton(
                        onClick = { viewModel.runFullSuite() },
                        enabled = !viewModel.isBusy
                    ) {
                        Icon(Icons.Filled.PlaylistPlay, contentDescription = "Run full suite")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer
                )
            )
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {

            if (viewModel.isRunningFullSuite) {
                Column(modifier = Modifier.fillMaxWidth().padding(16.dp, 8.dp)) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        viewModel.fullSuiteStatus,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }

            viewModel.lastStartupError?.let { message ->
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth().padding(16.dp, 4.dp)
                ) {
                    Text(
                        message,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text("Search configs...") },
                leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
                trailingIcon = {
                    if (query.isNotEmpty()) {
                        IconButton(onClick = { query = "" }) {
                            Icon(Icons.Filled.Close, contentDescription = "Clear")
                        }
                    }
                },
                singleLine = true,
                modifier = Modifier.fillMaxWidth().padding(16.dp, 8.dp)
            )

            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
            ) {
                CategoryFilter.values().forEach { option ->
                    FilterChip(
                        selected = filter == option,
                        onClick = { filter = option },
                        label = { Text(option.label) }
                    )
                }
            }

            val totalShown = filteredModels.size + filteredPipelines.size
            Text(
                "$totalShown item${if (totalShown == 1) "" else "s"}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(16.dp, 8.dp, 16.dp, 0.dp)
            )

            LazyColumn(
                contentPadding = PaddingValues(16.dp, 8.dp, 16.dp, 24.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.fillMaxSize()
            ) {
                items(filteredModels, key = { it.name }) { config ->
                    ConfigCard(
                        config = config,
                        outcome = viewModel.results[config.name],
                        isRunning = viewModel.runningName == config.name,
                        isBusy = viewModel.isBusy,
                        onRun = { viewModel.runConfig(config) }
                    )
                }
                items(filteredPipelines, key = { it.name }) { pipeline ->
                    PipelineCard(
                        pipeline = pipeline,
                        outcome = viewModel.results[pipeline.name],
                        isRunning = viewModel.runningName == pipeline.name,
                        isBusy = viewModel.isBusy,
                        onRun = { viewModel.runPipeline(pipeline) }
                    )
                }
            }
        }
    }
}

@Composable
private fun ConfigCard(
    config: ModelConfig,
    outcome: RunOutcome?,
    isRunning: Boolean,
    isBusy: Boolean,
    onRun: () -> Unit
) {
    val result = (outcome as? RunOutcome.Single)?.result
    val summary: List<String> = when {
        result == null -> emptyList()
        result.error != null -> listOf("Failed: ${result.error.lineSequence().first()}")
        else -> listOf(
            "p50 ${fmtMs(result.percentile(50.0))}  ·  p90 ${fmtMs(result.percentile(90.0))}",
            "PSS peak ${fmtMb(result.pssPeakDuringInferenceKb)}  ·  RSS peak ${fmtMbLong(result.rssPeakDuringInferenceKb)}"
        )
    }
    RunnableCard(
        title = config.name,
        chips = listOfNotNull(config.task, config.runtime, config.delegate),
        isRunning = isRunning,
        isBusy = isBusy,
        isError = result?.error != null,
        isSuccess = result != null && result.error == null,
        summaryLines = summary,
        onRun = onRun
    )
}

@Composable
private fun PipelineCard(
    pipeline: PipelineConfig,
    outcome: RunOutcome?,
    isRunning: Boolean,
    isBusy: Boolean,
    onRun: () -> Unit
) {
    val result = (outcome as? RunOutcome.Pipeline)?.result
    val summary: List<String> = when {
        result == null -> emptyList()
        result.error != null -> listOf("Failed: ${result.error.lineSequence().first()}")
        else -> {
            val detP50 = result.percentile(result.detectorLatenciesMs, 50.0)
            val ocrP50 = result.percentile(result.ocrLatenciesMs, 50.0)
            listOf(
                "detector ${fmtMs(detP50)}  +  ocr ${fmtMs(ocrP50)}  =  ${fmtMs(detP50 + ocrP50)} end-to-end",
                "peak PSS overall ${fmtMb(result.pssPeakOverallKb)}"
            )
        }
    }
    RunnableCard(
        title = pipeline.name,
        chips = listOf("pipeline"),
        subtitle = "det: ${pipeline.detectorConfigName}  →  ocr: ${pipeline.ocrConfigName}",
        isRunning = isRunning,
        isBusy = isBusy,
        isError = result?.error != null,
        isSuccess = result != null && result.error == null,
        summaryLines = summary,
        onRun = onRun
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RunnableCard(
    title: String,
    chips: List<String>,
    isRunning: Boolean,
    isBusy: Boolean,
    isError: Boolean,
    isSuccess: Boolean,
    summaryLines: List<String>,
    onRun: () -> Unit,
    subtitle: String? = null
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = when {
                isError -> MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.35f)
                else -> MaterialTheme.colorScheme.surfaceContainerLow
            }
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth().padding(14.dp)
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (isSuccess) {
                        Icon(
                            Icons.Filled.CheckCircle,
                            contentDescription = "Succeeded",
                            tint = Color(0xFF2E7D32),
                            modifier = Modifier.size(16.dp)
                        )
                        Box(modifier = Modifier.size(6.dp))
                    } else if (isError) {
                        Icon(
                            Icons.Filled.Error,
                            contentDescription = "Failed",
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(16.dp)
                        )
                        Box(modifier = Modifier.size(6.dp))
                    }
                    Text(
                        title,
                        fontWeight = FontWeight.Medium,
                        style = MaterialTheme.typography.bodyLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                // FlowRow (not a plain Row) so chips wrap onto a new line instead of
                // overflowing/getting cut off when they don't all fit on one line --
                // confirmed real on pipeline cards, whose detector/OCR names push chip
                // rows wider than a plain Row handles.
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                    modifier = Modifier.padding(top = 6.dp)
                ) {
                    chips.forEach { label ->
                        AssistChip(
                            onClick = {},
                            label = { Text(label, style = MaterialTheme.typography.labelSmall) },
                            colors = AssistChipDefaults.assistChipColors(
                                containerColor = MaterialTheme.colorScheme.secondaryContainer
                            )
                        )
                    }
                }

                // Long, variable-length free text (e.g. a pipeline's detector/OCR config
                // names) goes here, NOT as a chip -- chips are for short fixed tags, and
                // cramming long dynamic text into one caused exactly the overlapping/
                // cut-off look this replaced. Truncated with an ellipsis if still too
                // long for the card rather than wrapping awkwardly or overflowing.
                subtitle?.let {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.padding(top = 6.dp)
                    )
                }

                summaryLines.forEach { line ->
                    Text(
                        line,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (isError) MaterialTheme.colorScheme.error
                                else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }

            Box(modifier = Modifier.size(40.dp), contentAlignment = Alignment.Center) {
                if (isRunning) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
                } else {
                    IconButton(onClick = onRun, enabled = !isBusy) {
                        Icon(Icons.Filled.PlayArrow, contentDescription = "Run $title")
                    }
                }
            }
        }
    }
}

private fun fmtMs(ms: Double): String =
    if (ms < 0) "N/A" else "${(ms * 10).roundToInt() / 10.0} ms"

private fun fmtMb(kb: Int): String =
    if (kb < 0) "N/A" else "${(kb / 1024.0 * 10).roundToInt() / 10.0} MB"

private fun fmtMbLong(kb: Long): String =
    if (kb < 0) "N/A" else "${(kb / 1024.0 * 10).roundToInt() / 10.0} MB"