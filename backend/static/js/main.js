/**
 * Chefly — Global UI interactions
 * Dark mode, navbar, scroll reveal, lazy images, scroll-to-top
 */
(function () {
  'use strict';

  const THEME_KEY = 'chefly-theme';
  const html = document.documentElement;

  /* ── Theme ─────────────────────────────────────────────── */
  function getTheme() {
    return html.getAttribute('data-theme') || 'light';
  }

  function setTheme(theme) {
    html.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', theme === 'dark' ? '#0b0d12' : '#ff6b35');
  }

  function initTheme() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      setTheme(getTheme() === 'dark' ? 'light' : 'dark');
    });
  }

  /* ── Sticky navbar ─────────────────────────────────────── */
  function initHeader() {
    const header = document.getElementById('siteHeader');
    if (!header) return;

    const onScroll = function () {
      header.classList.toggle('is-scrolled', window.scrollY > 12);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* ── Mobile menu ───────────────────────────────────────── */
  function initMobileNav() {
    const toggle = document.getElementById('navToggle');
    const panel = document.getElementById('mobileNav');
    if (!toggle || !panel) return;

    const open = function () {
      panel.hidden = false;
      requestAnimationFrame(function () {
        panel.classList.add('is-open');
        toggle.classList.add('is-open');
        toggle.setAttribute('aria-expanded', 'true');
        toggle.setAttribute('aria-label', 'Close menu');
        document.body.style.overflow = 'hidden';
      });
    };

    const close = function () {
      panel.classList.remove('is-open');
      toggle.classList.remove('is-open');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-label', 'Open menu');
      document.body.style.overflow = '';
      setTimeout(function () {
        if (!panel.classList.contains('is-open')) panel.hidden = true;
      }, 280);
    };

    toggle.addEventListener('click', function () {
      panel.classList.contains('is-open') ? close() : open();
    });

    panel.querySelectorAll('.mobile-nav__link').forEach(function (link) {
      link.addEventListener('click', close);
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') close();
    });
  }

  /* ── Active nav link ───────────────────────────────────── */
  function initActiveNav() {
    const path = window.location.pathname.replace(/\/$/, '') || '/';
    const map = {
      '/': 'home',
      '/about': 'about',
      '/contact': 'contact',
      '/famous-dishes': 'famous_dishes',
      '/trained_dishes': 'trained_dishes',
      '/upload': 'upload_form',
      '/generate': 'generate_recipe',
      '/submit-recipe': 'submit_recipe',
      '/login': 'login',
      '/signup': 'signup',
      '/search': 'search',
      '/profile': 'profile'
    };

    let key = 'home';
    for (const [route, navKey] of Object.entries(map)) {
      if (path === route || (route !== '/' && path.startsWith(route))) {
        key = navKey;
        break;
      }
    }

    document.querySelectorAll('[data-nav]').forEach(function (el) {
      el.classList.toggle('is-active', el.getAttribute('data-nav') === key);
    });
  }

  /* ── Scroll reveal ─────────────────────────────────────── */
  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (!els.length) return;

    if (!('IntersectionObserver' in window)) {
      els.forEach(function (el) { el.classList.add('is-visible'); });
      return;
    }

    const io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );

    els.forEach(function (el) { io.observe(el); });
  }

  /* ── Lazy images ───────────────────────────────────────── */
  function initLazyImages() {
    const imgs = document.querySelectorAll('img[loading="lazy"]');
    imgs.forEach(function (img) {
      const done = function () {
        img.classList.add('is-loaded');
        const sk = img.parentElement && img.parentElement.querySelector('.img-skeleton');
        if (sk) sk.remove();
      };
      if (img.complete) done();
      else {
        img.addEventListener('load', done, { once: true });
        img.addEventListener('error', done, { once: true });
      }
    });
  }

  /* ── Scroll to top ─────────────────────────────────────── */
  function initScrollTop() {
    const btn = document.getElementById('scrollTopBtn');
    if (!btn) return;

    window.addEventListener('scroll', function () {
      btn.classList.toggle('is-visible', window.scrollY > 500);
    }, { passive: true });

    btn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ── Empty search guard ────────────────────────────────── */
  function initSearchForms() {
    document.querySelectorAll('form[role="search"], .navbar__search, .mobile-nav__search, .hero-search-form').forEach(function (form) {
      form.addEventListener('submit', function (e) {
        const input = form.querySelector('input[name="q"]');
        if (input && !input.value.trim()) e.preventDefault();
      });
    });
  }

  /* ── Newsletter (demo) ─────────────────────────────────── */
  function initNewsletter() {
    document.querySelectorAll('.newsletter-form').forEach(function (form) {
      form.addEventListener('submit', function (e) {
        e.preventDefault();
        const input = form.querySelector('input[type="email"]');
        if (!input || !input.value.trim()) return;
        const btn = form.querySelector('button[type="submit"]');
        const original = btn ? btn.innerHTML : '';
        if (btn) {
          btn.innerHTML = '<i class="fa-solid fa-check"></i> Subscribed!';
          btn.disabled = true;
        }
        input.value = '';
        setTimeout(function () {
          if (btn) {
            btn.innerHTML = original;
            btn.disabled = false;
          }
        }, 2800);
      });
    });
  }

  /* ── Boot ──────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initHeader();
    initMobileNav();
    initActiveNav();
    initReveal();
    initLazyImages();
    initScrollTop();
    initSearchForms();
    initNewsletter();
  });
})();
