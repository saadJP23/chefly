(function () {
  "use strict";

  function apiBaseUrl() {
    return (window.CHEFLY_CONFIG && window.CHEFLY_CONFIG.apiBaseUrl || "").replace(/\/$/, "");
  }

  window.cheflyApiUrl = function (path) {
    const cleanPath = path.startsWith("/") ? path : "/" + path;
    return apiBaseUrl() + cleanPath;
  };

  window.cheflyContactUrl = function () {
    const base = apiBaseUrl();
    return base ? base + "/contact" : null;
  };

  document.addEventListener("DOMContentLoaded", function () {
    const contactLinks = document.querySelectorAll('a[href="/contact"], a[href="/contact.html"]');
    const url = window.cheflyContactUrl();
    contactLinks.forEach(function (link) {
      if (url) {
        link.href = url;
      }
    });
  });
})();
