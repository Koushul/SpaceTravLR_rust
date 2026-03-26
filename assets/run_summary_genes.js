(function () {
  function readMetrics() {
    var el = document.getElementById("spacetravlr-gene-metrics");
    if (!el || !el.textContent) return [];
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return [];
    }
  }

  function fmtNum(x, d) {
    if (x === null || x === undefined || typeof x !== "number" || !isFinite(x)) return "—";
    return x.toFixed(d);
  }

  function fmtPct(x) {
    if (x === null || x === undefined || typeof x !== "number" || !isFinite(x)) return "—";
    return (100 * x).toFixed(1) + "%";
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  var data = readMetrics();
  var root = document.getElementById("gene-metrics-root");
  if (!root) return;

  var sortState = { col: "mean_lasso_r2", dir: -1 };

  function cmp(a, b, key) {
    var va = a[key];
    var vb = b[key];
    if (typeof va === "number" && typeof vb === "number") return va - vb;
    return String(va).localeCompare(String(vb));
  }

  function sortedRows(filter) {
    var q = (filter || "").trim().toLowerCase();
    var rows = data.filter(function (g) {
      return !q || g.gene.toLowerCase().indexOf(q) !== -1;
    });
    var key = sortState.col;
    var dir = sortState.dir;
    rows.sort(function (a, b) {
      var c = cmp(a, b, key);
      return dir * c;
    });
    return rows;
  }

  function gateHtml(gate) {
    if (!gate) return "";
    var keys = Object.keys(gate).sort();
    if (!keys.length) return "";
    var parts = keys.map(function (k) {
      return "<dt>" + esc(k) + "</dt><dd>" + esc(gate[k]) + "</dd>";
    });
    return '<dl class="gene-gate-dl">' + parts.join("") + "</dl>";
  }

  function clusterTable(clusters) {
    var h =
      "<table class=\"gene-detail-table\"><thead><tr>" +
      "<th>cluster</th><th>n cells</th><th>n mod</th><th>LASSO R²</th><th>train MSE</th>" +
      "<th>FISTA iters</th><th>converged</th><th>CNN epochs</th><th>CNN MSE 1st</th><th>CNN MSE last</th>" +
      "</tr></thead><tbody>";
    for (var i = 0; i < clusters.length; i++) {
      var c = clusters[i];
      h +=
        "<tr><td>" +
        c.cluster_id +
        "</td><td>" +
        c.n_cells +
        "</td><td>" +
        c.n_modulators +
        "</td><td>" +
        fmtNum(c.lasso_r2, 4) +
        "</td><td>" +
        fmtNum(c.lasso_train_mse, 6) +
        "</td><td>" +
        c.lasso_fista_iters +
        "</td><td>" +
        (c.lasso_converged ? "yes" : "no") +
        "</td><td>" +
        c.cnn_epochs_ran +
        "</td><td>" +
        (c.cnn_mse_first != null ? fmtNum(c.cnn_mse_first, 6) : "NA") +
        "</td><td>" +
        (c.cnn_mse_last != null ? fmtNum(c.cnn_mse_last, 6) : "NA") +
        "</td></tr>";
    }
    return h + "</tbody></table>";
  }

  function render() {
    var filterEl = document.getElementById("gene-filter-input");
    var filter = filterEl ? filterEl.value : "";
    var rows = sortedRows(filter);

    var countEl = document.getElementById("gene-metrics-count");
    if (countEl) {
      countEl.textContent =
        rows.length === data.length
          ? data.length + " genes (from log/)"
          : rows.length + " matching · " + data.length + " total";
    }

    var thead =
      "<thead><tr>" +
      '<th class="sortable" data-sort="gene">Gene</th>' +
      '<th class="sortable num" data-sort="mean_lasso_r2">Mean R²</th>' +
      '<th class="sortable num" data-sort="min_lasso_r2">Min R²</th>' +
      '<th class="sortable num" data-sort="max_lasso_r2">Max R²</th>' +
      '<th class="sortable num" data-sort="frac_lasso_converged">LASSO conv.</th>' +
      '<th class="sortable num" data-sort="sum_cnn_epochs_ran">Σ CNN epochs</th>' +
      '<th class="sortable" data-sort="seed_only">Seed-only</th>' +
      '<th class="sortable" data-sort="per_cell_cnn_export">Full CNN</th>' +
      '<th class="sortable num" data-sort="n_clusters">Clusters</th>' +
      "<th></th>" +
      "</tr></thead>";

    var body = "";
    for (var i = 0; i < rows.length; i++) {
      var g = rows[i];
      var rid = "gene-exp-" + i;
      body +=
        "<tr class=\"gene-row\" data-exp=\"" +
        esc(rid) +
        "\">" +
        "<td><strong>" +
        esc(g.gene) +
        "</strong></td>" +
        "<td class=\"num\">" +
        fmtNum(g.mean_lasso_r2, 4) +
        "</td>" +
        "<td class=\"num\">" +
        fmtNum(g.min_lasso_r2, 4) +
        "</td>" +
        "<td class=\"num\">" +
        fmtNum(g.max_lasso_r2, 4) +
        "</td>" +
        "<td class=\"num\">" +
        fmtPct(g.frac_lasso_converged) +
        "</td>" +
        "<td class=\"num\">" +
        g.sum_cnn_epochs_ran +
        "</td>" +
        "<td>" +
        (g.seed_only ? "yes" : "no") +
        "</td>" +
        "<td>" +
        (g.per_cell_cnn_export ? "yes" : "no") +
        "</td>" +
        "<td class=\"num\">" +
        g.n_clusters +
        "</td>" +
        '<td class="exp-hint">▸ details</td>' +
        "</tr>" +
        "<tr class=\"gene-exp-row\" id=\"" +
        rid +
        "\" hidden><td colspan=\"10\">" +
        '<div class="gene-detail-inner">' +
        "<p class=\"gene-detail-meta\">LR " +
        fmtNum(g.learning_rate, 6) +
        " · LASSO max iter " +
        g.lasso_n_iter_max +
        " · tol " +
        fmtNum(g.lasso_tol, 6) +
        " · CNN epochs (config) " +
        g.cnn_epochs_config +
        "</p>" +
        gateHtml(g.gate) +
        clusterTable(g.clusters) +
        "</div></td></tr>";
    }

    var wrap = document.getElementById("gene-table-wrap");
    if (wrap) {
      if (data.length === 0) {
        wrap.innerHTML =
          '<p class="gene-empty">No per-gene training logs found under <code>log/*.log</code>. ' +
          "Run training with log export enabled, or open this report from the training output directory.</p>";
      } else {
        wrap.innerHTML = '<table class="gene-metrics-table" id="gene-metrics-table">' + thead + "<tbody>" + body + "</tbody></table>";
        var table = document.getElementById("gene-metrics-table");
        if (table) {
          table.querySelectorAll("th.sortable").forEach(function (th) {
            th.addEventListener("click", function () {
              var k = th.getAttribute("data-sort");
              if (sortState.col === k) sortState.dir = -sortState.dir;
              else {
                sortState.col = k;
                sortState.dir = k === "gene" ? 1 : -1;
              }
              render();
            });
          });
          table.querySelectorAll("tr.gene-row").forEach(function (tr) {
            tr.addEventListener("click", function () {
              var id = tr.getAttribute("data-exp");
              var exp = document.getElementById(id);
              if (!exp) return;
              var open = !exp.hidden;
              table.querySelectorAll("tr.gene-exp-row").forEach(function (r) {
                r.hidden = true;
              });
              table.querySelectorAll("tr.gene-row").forEach(function (r) {
                r.classList.remove("gene-row-open");
              });
              if (!open) {
                exp.hidden = false;
                tr.classList.add("gene-row-open");
              }
            });
          });
        }
      }
    }
  }

  var filterInput = document.getElementById("gene-filter-input");
  if (filterInput) {
    filterInput.addEventListener("input", function () {
      render();
    });
  }

  render();
})();
