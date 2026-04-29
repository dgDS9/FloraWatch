import { createContext, useContext, useState, useCallback, useMemo } from 'react';
import en from './en';
import de from './de';
import es from './es';

const LANGS = { en, de, es };
const LANG_LABELS = { en: 'EN', de: 'DE', es: 'ES' };
const STORAGE_KEY = 'florawatch_lang';

function getBrowserLang() {
  const nav = navigator.language?.slice(0, 2);
  return LANGS[nav] ? nav : 'en';
}

function getInitialLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && LANGS[saved]) return saved;
  } catch {}
  return getBrowserLang();
}

const I18nContext = createContext();

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(getInitialLang);

  const setLang = useCallback((l) => {
    setLangState(l);
    try { localStorage.setItem(STORAGE_KEY, l); } catch {}
  }, []);

  const t = useCallback((key) => LANGS[lang]?.[key] || LANGS.en[key] || key, [lang]);

  const value = useMemo(() => ({ lang, setLang, t, langs: Object.keys(LANGS), langLabels: LANG_LABELS }), [lang, setLang, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
