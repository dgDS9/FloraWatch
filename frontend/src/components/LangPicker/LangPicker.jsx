import { useI18n } from '../../i18n/I18nContext';
import './LangPicker.css';

export default function LangPicker() {
  const { lang, setLang, langs, langLabels } = useI18n();

  return (
    <div className="lang-picker">
      {langs.map((l) => (
        <button
          key={l}
          className={`lang-btn${l === lang ? ' lang-btn--active' : ''}`}
          onClick={() => setLang(l)}
        >
          {langLabels[l]}
        </button>
      ))}
    </div>
  );
}
