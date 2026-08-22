import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const sourcePath = join(
  root,
  "src/shaders/sylva-living-world/sources/inner-green-3d.html"
);
const runtimePath = join(
  root,
  "src/shaders/sylva-living-world/sources/inner-green-assets/three.min.js"
);
const outputPath = join(root, "sylva-tree-scene.html");

const innerGreenSource = readFileSync(sourcePath, "utf8");
const threeRuntime = readFileSync(runtimePath, "utf8");

const presentationStart = innerGreenSource.indexOf('<main class="hero" id="hero">');
const runtimeStart = innerGreenSource.indexOf(
  '<script src="inner-green-assets/three.min.js"></script>'
);

if (presentationStart < 0 || runtimeStart < 0 || runtimeStart <= presentationStart) {
  throw new Error("Could not isolate the Sylva Three.js scene from inner-green-3d.html.");
}

const sceneOnlyMarkup =
  '<main class="hero is-ready" id="hero"><canvas id="scene" role="img" aria-label="Sylva Living Green"></canvas><div class="stage" id="stage" aria-hidden="true"></div></main>';

const sceneOnlyStyle = `<style data-threeui-sylva-scene>
html,body{width:100%!important;height:100%!important;min-height:0!important;margin:0!important;overflow:hidden!important;background:transparent!important}
body{position:relative!important}
.hero{position:relative!important;width:100%!important;height:100%!important;min-height:0!important;overflow:hidden!important;background:transparent!important}
.hero::after{display:none!important}
.stage{position:absolute!important;inset:0!important;margin:0!important;width:100%!important;height:100%!important}
#scene{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;pointer-events:auto!important;opacity:1!important;z-index:1!important;background:transparent!important;display:block!important}
canvas#scene{background:transparent!important}
</style>`;

let output = `${innerGreenSource.slice(0, presentationStart)}${sceneOnlyMarkup}

${innerGreenSource.slice(runtimeStart)}`
  .replace("</head>", `${sceneOnlyStyle}</head>`)
  .replace(
    '<script src="inner-green-assets/three.min.js"></script>',
    `<script>${threeRuntime}</script>`
  )
  .replace(
    "if (!REDUCED && !document.hidden) { uScanOn.value = 1; uScanR.value = 0; scanning = true; }",
    "uScanOn.value = 0; uScanR.value = scanMax; scanning = false;"
  )
  .replace(
    "setTimeout(function () { document.body.classList.add('intro-done'); }, REDUCED ? 0 : 2900);",
    "document.body.classList.add('intro-done');"
  )
  .replace(
    "var NARROW   = window.matchMedia('(max-width: 900px)');",
    "var NARROW   = { matches: false, addEventListener: function () {} };"
  )
  .replace(
    "uHaze:    { value: 0.14 },",
    "uHaze:    { value: 0.0 },"
  )
  .replace(
    "haze: 0.15, fog: 0.0, alpha: 1.0, order: 2,",
    "haze: 0.0, fog: 0.0, alpha: 1.0, order: 2,"
  )
  .replace(
    "haze: 0.16, fog: 0.26, alpha: 1.0, order: 0,",
    "haze: 0.0, fog: 0.0, alpha: 1.0, order: 0,"
  )
  .replace(
    "map: radialTexture(256, [[0, 'rgba(12,16,10,0.62)'], [0.45, 'rgba(12,16,10,0.26)'], [1, 'rgba(12,16,10,0)']]),",
    "map: radialTexture(256, [[0, 'rgba(12,16,10,0)'], [1, 'rgba(12,16,10,0)']]),"
  )
  .replace(
    "map: radialTexture(256, [[0, 'rgba(226,236,212,0.30)'], [0.42, 'rgba(214,226,200,0.10)'], [1, 'rgba(214,226,200,0)']]),",
    "map: radialTexture(256, [[0, 'rgba(226,236,212,0)'], [1, 'rgba(214,226,200,0)']]),"
  )
  .replace(
    "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: !small });",
    "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: false, premultipliedAlpha: false, powerPreference: 'high-performance', stencil: false });"
  )
  .replace(
    "renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, small ? 1.6 : 2));",
    "renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));"
  )
  .replace(
    "renderer.setClearColor(0x000000, 0);",
    "renderer.setClearColor(0x000000, 0); canvas.style.background = 'transparent';"
  )
  .replace(
    "if (!small) {\n      bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n      bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);\n    }",
    "bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n    bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);"
  )
  .replaceAll("wire: true,", "wire: false,")
  .replace("scene.add(shadowMesh);", "")
  .replace("scene.add(glowMesh);", "")
  .replace(
    "(function loop() { requestAnimationFrame(loop); tick(); })();",
    `(function loop() {
      requestAnimationFrame(loop);
      if (document.hidden || window.__sylvaVisible === false) return;
      tick();
    })();`
  )
  .replace(
    "if (renderer && clock) renderFrame();",
    "if (renderer && clock && !(REDUCED && frames > 1)) renderFrame();"
  )
  .replace(
    "startPortalReveal();\n    initDock();",
    ""
  )
  .replace(
    "src:url('inner-green-assets/lexend-latin.woff2') format('woff2');",
    "src:local('Lexend');"
  );

writeFileSync(outputPath, output);
console.log(`wrote ${outputPath} (${output.length} bytes)`);
