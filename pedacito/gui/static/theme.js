// Applied before first paint so a saved theme/layout does not flash.
// The default scheme is "slate" (GitHub Dark), carried by :root with no
// attribute; the old alias names are gone, so an unknown saved value simply
// falls back to that default.
(function () {
  try {
    var t = localStorage.getItem('pedacito_theme');
    if (t && t !== 'slate') document.documentElement.setAttribute('data-theme', t);
    var w = JSON.parse(localStorage.getItem('pedacito_layout') || '{}');
    var r = document.documentElement.style;
    if (w.left) r.setProperty('--left', w.left + 'px');
    if (w.right) r.setProperty('--right', w.right + 'px');
    if (w.term) r.setProperty('--term-h', w.term + 'px');
    var ch = parseInt(localStorage.getItem('pedacito_composer_height') || '0', 10);
    if (ch) r.setProperty('--composer-h', Math.max(64, ch) + 'px');
  } catch (e) {}
})();
