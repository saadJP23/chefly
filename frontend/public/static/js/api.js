(function () {
  "use strict";

  function apiBaseUrl() {
    return (window.CHEFLY_CONFIG && window.CHEFLY_CONFIG.apiBaseUrl || "").replace(/\/$/, "");
  }

  window.cheflyApiUrl = function (path) {
    const cleanPath = path.startsWith("/") ? path : "/" + path;
    return apiBaseUrl() + cleanPath;
  };
})();
