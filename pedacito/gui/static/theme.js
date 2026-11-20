// Applied before first paint so a saved theme/layout does not flash.
// The scheme marker (localStorage pedacito_theme) is either one of the nine
// CSS-native gui_theme schemes -- painted by its crafted app.css block via
// the data-theme attribute -- or an engine palette name (a themes.json
// preset/custom), painted by the cached ten-token inline properties. A
// CSS-native marker, or the server-injected gui_theme attribute when no
// marker exists, must never be overpainted by a stale engine cache.
(function () {
  var CSS_SCHEMES = ['slate', 'light', 'daylight', 'nord', 'midnight',
    'jazz_solo', 'trapper_keeper', 'memphis_arcade', 'windbreaker'];
  try {
    var t = localStorage.getItem('pedacito_theme');
    var attr = document.documentElement.getAttribute('data-theme');
    var cssNative = t && CSS_SCHEMES.indexOf(t) >= 0;
    var serverCss = !t && attr && CSS_SCHEMES.indexOf(attr) >= 0;
    if (cssNative) {
      if (t === 'slate') document.documentElement.removeAttribute('data-theme');
      else document.documentElement.setAttribute('data-theme', t);
    } else if (t) {
      // Engine palette: the inline cache below paints it, no CSS block.
      document.documentElement.removeAttribute('data-theme');
    }
    var r = document.documentElement.style;
    if (!cssNative && !serverCss) {
      // Unified theme engine: the last applied ten-token palette is cached as
      // fully resolved CSS properties, so a reload paints in the user's colors
      // before /api/themes answers. app.js refreshes the cache after a switch.
      var cached = JSON.parse(localStorage.getItem('pedacito_theme_css') || '{}');
      Object.keys(cached).forEach(function (k) {
        if (cached[k]) r.setProperty(k, cached[k]);
      });
    }
    var w = JSON.parse(localStorage.getItem('pedacito_layout') || '{}');
    if (w.left) r.setProperty('--left', w.left + 'px');
    if (w.right) r.setProperty('--right', w.right + 'px');
    if (w.term) r.setProperty('--term-h', w.term + 'px');
    var ch = parseInt(localStorage.getItem('pedacito_composer_height') || '0', 10);
    if (ch) r.setProperty('--composer-h', Math.max(64, ch) + 'px');
  } catch (e) {}
})();
