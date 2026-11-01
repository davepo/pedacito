// Applied before first paint so a saved theme does not flash.
(function () {
  try {
    var t = localStorage.getItem('pedacito_theme');
    if (t && t !== 'parchment') document.documentElement.setAttribute('data-theme', t);
    var w = JSON.parse(localStorage.getItem('pedacito_layout') || '{}');
    var r = document.documentElement.style;
    if (w.left) r.setProperty('--left', w.left + 'px');
    if (w.right) r.setProperty('--right', w.right + 'px');
    if (w.term) r.setProperty('--term-h', w.term + 'px');
  } catch (e) {}
})();
