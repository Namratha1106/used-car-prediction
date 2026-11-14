(() => {
  const palette = {
    navy: "#17243a",
    muted: "#8792a2",
    orange: "#f47b3f",
    blue: "#5277d5",
    purple: "#8e79d6",
    green: "#36a889",
    grid: "rgba(28, 43, 64, .07)",
  };
  const modelColors = [palette.blue, palette.orange, palette.green];

  const readJson = (id) => {
    const node = document.getElementById(id);
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      console.error(`Could not read chart data from ${id}.`, error);
      return null;
    }
  };

  const money = (value) => `₹${Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 650 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#142238",
        titleFont: { family: "DM Sans", weight: "600" },
        bodyFont: { family: "DM Sans" },
        padding: 11,
        cornerRadius: 9,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        border: { display: false },
        ticks: { color: palette.muted, font: { family: "DM Sans", size: 10 }, maxRotation: 0 },
      },
      y: {
        grid: { color: palette.grid },
        border: { display: false, dash: [3, 5] },
        ticks: { color: palette.muted, font: { family: "DM Sans", size: 10 } },
      },
    },
  };

  const drawBar = (id, labels, values, options = {}) => {
    const canvas = document.querySelector(`[data-chart-id="${id}"]`);
    if (!canvas || typeof Chart === "undefined" || !labels?.length) return;
    const horizontal = options.horizontal === true;
    const config = {
      type: options.type || "bar",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: options.colors || palette.blue,
          borderRadius: options.radius ?? 6,
          borderSkipped: false,
          maxBarThickness: options.maxBarThickness || 34,
          borderWidth: 0,
          ...(options.extraDataset || {}),
        }],
      },
      options: {
        ...commonOptions,
        indexAxis: horizontal ? "y" : "x",
        scales: {
          ...commonOptions.scales,
          ...(horizontal ? {
            x: { ...commonOptions.scales.y, beginAtZero: true },
            y: { ...commonOptions.scales.x, grid: { display: false }, ticks: { ...commonOptions.scales.x.ticks, autoSkip: false } },
          } : {}),
          ...(options.scales || {}),
        },
        plugins: {
          ...commonOptions.plugins,
          tooltip: {
            ...commonOptions.plugins.tooltip,
            callbacks: options.tooltip || {},
          },
        },
      },
    };
    new Chart(canvas, config);
  };

  const performance = readJson("performance-data");
  if (performance && typeof Chart !== "undefined") {
    const entries = Object.entries(performance.metrics);
    const labels = entries.map(([, metric]) => metric.name.replace(" Regressor", ""));
    drawBar(
      "rmse-chart",
      labels,
      entries.map(([, metric]) => metric.rmse),
      { colors: modelColors, tooltip: { label: (ctx) => ` RMSE: ${money(ctx.raw)}` } },
    );
    drawBar(
      "r2-chart",
      labels,
      entries.map(([, metric]) => metric.r2),
      {
        colors: modelColors,
        scales: { y: { ...commonOptions.scales.y, suggestedMin: 0, suggestedMax: 1 } },
        tooltip: { label: (ctx) => ` R²: ${Number(ctx.raw).toFixed(4)}` },
      },
    );
    const scatterCanvas = document.querySelector('[data-chart-id="scatter-chart"]');
    if (scatterCanvas && performance.actual_vs_predicted) {
      const points = performance.actual_vs_predicted.actual.map((actual, index) => ({
        x: actual,
        y: performance.actual_vs_predicted.predicted[index],
      }));
      new Chart(scatterCanvas, {
        type: "scatter",
        data: { datasets: [{ label: "Held-out predictions", data: points, backgroundColor: "rgba(244, 123, 63, .58)", borderColor: palette.orange, pointRadius: 4, pointHoverRadius: 6 }] },
        options: {
          ...commonOptions,
          scales: {
            x: { ...commonOptions.scales.x, title: { display: true, text: "Actual price", color: palette.muted }, ticks: { ...commonOptions.scales.x.ticks, callback: (v) => money(v) } },
            y: { ...commonOptions.scales.y, title: { display: true, text: "Predicted price", color: palette.muted }, ticks: { ...commonOptions.scales.y.ticks, callback: (v) => money(v) } },
          },
          plugins: {
            ...commonOptions.plugins,
            tooltip: {
              ...commonOptions.plugins.tooltip,
              callbacks: {
                label: (ctx) => ` Actual ${money(ctx.raw.x)} · Predicted ${money(ctx.raw.y)}`,
              },
            },
          },
        },
      });
    }
    const importance = performance.feature_importance;
    drawBar("importance-chart", importance?.labels, importance?.values, {
      horizontal: true,
      colors: "rgba(82, 119, 213, .78)",
      maxBarThickness: 15,
      tooltip: { label: (ctx) => ` Importance: ${Number(ctx.raw).toFixed(4)}` },
    });
  }

  const analysis = readJson("analysis-data");
  if (analysis && typeof Chart !== "undefined") {
    drawBar("analysis-price", analysis.price?.labels, analysis.price?.values, {
      colors: "rgba(244, 123, 63, .78)",
      tooltip: { label: (ctx) => ` ${ctx.raw.toLocaleString()} listings` },
    });
    if (analysis.fuel?.labels?.length) {
      const canvas = document.querySelector('[data-chart-id="analysis-fuel"]');
      if (canvas) new Chart(canvas, {
        type: "doughnut",
        data: {
          labels: analysis.fuel.labels,
          datasets: [{ data: analysis.fuel.values, backgroundColor: [palette.orange, palette.blue, palette.green, palette.purple, "#e5b64b", "#58a6b8"], borderWidth: 3, borderColor: "#fff", hoverOffset: 5 }],
        },
        options: { responsive: true, maintainAspectRatio: false, cutout: "68%", plugins: { ...commonOptions.plugins, legend: { display: true, position: "bottom", labels: { usePointStyle: true, boxWidth: 7, padding: 17, color: palette.muted, font: { family: "DM Sans", size: 11 } } } } },
      });
    }
    drawBar("analysis-brand", analysis.brand?.labels, analysis.brand?.values, {
      horizontal: true,
      colors: "rgba(82, 119, 213, .78)",
      maxBarThickness: 16,
      tooltip: { label: (ctx) => ` ${ctx.raw.toLocaleString()} listings` },
    });
    drawBar("analysis-year", analysis.year?.labels, analysis.year?.values, {
      colors: "rgba(54, 168, 137, .75)",
      maxBarThickness: 28,
      tooltip: { label: (ctx) => ` ${ctx.raw.toLocaleString()} listings` },
    });
    drawBar("analysis-distance", analysis.distance?.labels, analysis.distance?.values, {
      colors: "rgba(142, 121, 214, .75)",
      maxBarThickness: 28,
      tooltip: { label: (ctx) => ` ${ctx.raw.toLocaleString()} listings` },
    });
  }

  const homeData = readJson("home-chart-data");
  drawBar("home-price", homeData?.labels, homeData?.values, {
    colors: "rgba(244, 123, 63, .76)",
    maxBarThickness: 22,
    tooltip: { label: (ctx) => ` ${ctx.raw.toLocaleString()} listings` },
  });

  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");
  menuToggle?.addEventListener("click", () => sidebar?.classList.toggle("sidebar-open"));
  document.addEventListener("click", (event) => {
    if (sidebar?.classList.contains("sidebar-open") &&
        !sidebar.contains(event.target) && !menuToggle?.contains(event.target)) {
      sidebar.classList.remove("sidebar-open");
    }
  });

  const form = document.querySelector("[data-prediction-form]");
  const errorBox = document.getElementById("predictionError");
  const resultCard = document.getElementById("predictionResult");
  const predictButton = document.getElementById("predictButton");
  form?.addEventListener("reset", () => {
    if (errorBox) {
      errorBox.hidden = true;
      errorBox.textContent = "";
    }
    resultCard?.classList.remove("has-result");
    resultCard?.querySelector(".result-placeholder")?.removeAttribute("hidden");
    resultCard?.querySelector(".result-success")?.setAttribute("hidden", "");
  });
  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    if (errorBox) errorBox.hidden = true;
    predictButton.disabled = true;
    predictButton.innerHTML = '<span class="button-spinner"></span> Calculating...';
    const fields = Object.fromEntries(
      [...form.querySelectorAll("[name]")].filter((field) => field.name !== "model")
        .map((field) => [field.name, field.value]),
    );
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: document.getElementById("modelSelect").value, features: fields }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The prediction request failed.");
      document.getElementById("predictedPrice").textContent = result.formatted_price;
      document.getElementById("predictedModel").textContent = result.model_name;
      resultCard?.classList.add("has-result");
      resultCard?.querySelector(".result-placeholder")?.setAttribute("hidden", "");
      resultCard?.querySelector(".result-success")?.removeAttribute("hidden");
    } catch (error) {
      if (errorBox) {
        errorBox.textContent = error.message || "Unable to generate a prediction.";
        errorBox.hidden = false;
      }
    } finally {
      predictButton.disabled = false;
      predictButton.innerHTML = 'Generate estimate <i class="bi bi-arrow-right"></i>';
    }
  });

  const clearButton = document.getElementById("clearHistory");
  clearButton?.addEventListener("click", async () => {
    if (!window.confirm("Clear all saved prediction history? This cannot be undone.")) return;
    clearButton.disabled = true;
    try {
      const response = await fetch("/api/history", { method: "DELETE" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Could not clear prediction history.");
      window.location.reload();
    } catch (error) {
      const toast = document.getElementById("historyToast");
      if (toast) {
        toast.textContent = error.message;
        toast.hidden = false;
      }
      clearButton.disabled = false;
    }
  });
})();
