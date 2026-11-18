// Applied before first paint so a saved theme/layout does not flash.
// The default scheme is "slate" (GitHub Dark), carried by :root with no
// attribute; the old alias names are gone, so an unknown saved value simply
// falls back to that default.
(function () {
  try {
    var t = localStorage.getItem('pedacito_theme');
    if (t && t !== 'slate') document.documentElement.setAttribute('data-theme', t);
    var r = document.documentElement.style;
    // Unified theme engine: the last applied ten-token palette is cached as
    // fully resolved CSS properties, so a reload paints in the user's colors
    // before /api/themes answers. app.js refreshes the cache after a switch.
    var cached = JSON.parse(localStorage.getItem('pedacito_theme_css') || '{}');
    Object.keys(cached).forEach(function (k) {
      if (cached[k]) r.setProperty(k, cached[k]);
    });
    var w = JSON.parse(localStorage.getItem('pedacito_layout') || '{}');
    if (w.left) r.setProperty('--left', w.left + 'px');
    if (w.right) r.setProperty('--right', w.right + 'px');
    if (w.term) r.setProperty('--term-h', w.term + 'px');
    var ch = parseInt(localStorage.getItem('pedacito_composer_height') || '0', 10);
    if (ch) r.setProperty('--composer-h', Math.max(64, ch) + 'px');
  } catch (e) {}
})();
