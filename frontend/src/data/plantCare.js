/**
 * Care information for the 23 species the model can identify.
 * Keyed by the scientific name (label) returned by the backend.
 * Text fields use { en, de, es } for i18n.
 */
const PLANT_CARE = {
  'Monstera deliciosa': {
    commonName: { en: 'Swiss Cheese Plant', de: 'Fensterblatt', es: 'Costilla de Adán' },
    water: {
      en: 'Moderate — let top 2–3 cm of soil dry out between waterings.',
      de: 'Mäßig — obere 2–3 cm Erde zwischen dem Gießen trocknen lassen.',
      es: 'Moderado — deja secar los 2–3 cm superiores entre riegos.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. Avoid direct sun.',
      de: 'Helles indirektes Licht. Direkte Sonne vermeiden.',
      es: 'Luz indirecta brillante. Evitar sol directo.',
    },
    toxicity: 'toxic',
  },
  'Epipremnum aureum': {
    commonName: { en: 'Golden Pothos', de: 'Goldene Efeutute', es: 'Potus dorado' },
    water: {
      en: 'Moderate — water when top soil feels dry.',
      de: 'Mäßig — gießen, wenn die obere Erde trocken ist.',
      es: 'Moderado — regar cuando la tierra superficial esté seca.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Low to bright indirect light. Very adaptable.',
      de: 'Wenig bis helles indirektes Licht. Sehr anpassungsfähig.',
      es: 'Luz indirecta baja a brillante. Muy adaptable.',
    },
    toxicity: 'toxic',
  },
  'Spathiphyllum wallisii': {
    commonName: { en: 'Peace Lily', de: 'Friedenslilie', es: 'Lirio de la paz' },
    water: {
      en: 'Keep soil consistently moist but not soggy.',
      de: 'Erde gleichmäßig feucht halten, aber nicht nass.',
      es: 'Mantener la tierra húmeda pero no empapada.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Low to medium indirect light. Tolerates shade.',
      de: 'Wenig bis mittleres indirektes Licht. Verträgt Schatten.',
      es: 'Luz indirecta baja a media. Tolera la sombra.',
    },
    toxicity: 'toxic',
  },
  'Sansevieria trifasciata': {
    commonName: { en: 'Snake Plant', de: 'Bogenhanf', es: 'Lengua de suegra' },
    water: {
      en: 'Sparingly — let soil dry out completely between waterings.',
      de: 'Sparsam — Erde zwischen dem Gießen vollständig trocknen lassen.',
      es: 'Con moderación — dejar secar completamente entre riegos.',
    },
    frequency: {
      en: 'Every 14–21 days',
      de: 'Alle 14–21 Tage',
      es: 'Cada 14–21 días',
    },
    sunlight: {
      en: 'Low to bright indirect light. Tolerates low light well.',
      de: 'Wenig bis helles indirektes Licht. Verträgt wenig Licht gut.',
      es: 'Luz indirecta baja a brillante. Tolera bien la poca luz.',
    },
    toxicity: 'mildlyToxic',
  },
  'Chlorophytum comosum': {
    commonName: { en: 'Spider Plant', de: 'Grünlilie', es: 'Cinta' },
    water: {
      en: 'Moderate — keep soil lightly moist.',
      de: 'Mäßig — Erde leicht feucht halten.',
      es: 'Moderado — mantener la tierra ligeramente húmeda.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. Avoid harsh direct sun.',
      de: 'Helles indirektes Licht. Starke direkte Sonne vermeiden.',
      es: 'Luz indirecta brillante. Evitar sol directo fuerte.',
    },
    toxicity: 'safe',
  },
  'Ficus elastica': {
    commonName: { en: 'Rubber Plant', de: 'Gummibaum', es: 'Árbol del caucho' },
    water: {
      en: 'Moderate — let top soil dry between waterings.',
      de: 'Mäßig — obere Erde zwischen dem Gießen trocknen lassen.',
      es: 'Moderado — dejar secar la superficie entre riegos.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. Some morning sun is fine.',
      de: 'Helles indirektes Licht. Etwas Morgensonne ist in Ordnung.',
      es: 'Luz indirecta brillante. Algo de sol matutino está bien.',
    },
    toxicity: 'mildlyToxic',
  },
  'Ficus benjamina': {
    commonName: { en: 'Weeping Fig', de: 'Birkenfeige', es: 'Ficus llorón' },
    water: {
      en: 'Moderate — keep soil evenly moist.',
      de: 'Mäßig — Erde gleichmäßig feucht halten.',
      es: 'Moderado — mantener la tierra uniformemente húmeda.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Bright indirect light. Sensitive to moving.',
      de: 'Helles indirektes Licht. Empfindlich beim Umstellen.',
      es: 'Luz indirecta brillante. Sensible a los cambios de lugar.',
    },
    toxicity: 'mildlyToxic',
  },
  'Dieffenbachia seguine': {
    commonName: { en: 'Dumb Cane', de: 'Dieffenbachie', es: 'Caña muda' },
    water: {
      en: 'Moderate — keep soil moist but not waterlogged.',
      de: 'Mäßig — Erde feucht halten, aber keine Staunässe.',
      es: 'Moderado — mantener húmeda sin encharcar.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Medium to bright indirect light.',
      de: 'Mittleres bis helles indirektes Licht.',
      es: 'Luz indirecta media a brillante.',
    },
    toxicity: 'toxic',
  },
  'Alocasia amazonica': {
    commonName: { en: 'Elephant Ear', de: 'Elefantenohr', es: 'Oreja de elefante' },
    water: {
      en: 'Keep soil moist but well-draining.',
      de: 'Erde feucht halten, aber gut durchlässig.',
      es: 'Mantener húmeda con buen drenaje.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Bright indirect light. High humidity preferred.',
      de: 'Helles indirektes Licht. Hohe Luftfeuchtigkeit bevorzugt.',
      es: 'Luz indirecta brillante. Prefiere alta humedad.',
    },
    toxicity: 'toxic',
  },
  'Anthurium andraeanum': {
    commonName: { en: 'Flamingo Flower', de: 'Flamingoblume', es: 'Flor de flamingo' },
    water: {
      en: 'Moderate — let top soil dry slightly between waterings.',
      de: 'Mäßig — obere Erde leicht trocknen lassen.',
      es: 'Moderado — dejar secar ligeramente la superficie.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. No direct sun.',
      de: 'Helles indirektes Licht. Keine direkte Sonne.',
      es: 'Luz indirecta brillante. Sin sol directo.',
    },
    toxicity: 'toxic',
  },
  'Philodendron hederaceum': {
    commonName: { en: 'Heartleaf Philodendron', de: 'Herzblatt-Philodendron', es: 'Filodendro hoja corazón' },
    water: {
      en: 'Moderate — let top 2 cm dry out.',
      de: 'Mäßig — obere 2 cm trocknen lassen.',
      es: 'Moderado — dejar secar los 2 cm superiores.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Low to bright indirect light.',
      de: 'Wenig bis helles indirektes Licht.',
      es: 'Luz indirecta baja a brillante.',
    },
    toxicity: 'toxic',
  },
  'Tradescantia zebrina': {
    commonName: { en: 'Wandering Jew', de: 'Dreimasterblume', es: 'Tradescantia cebrina' },
    water: {
      en: 'Keep soil moist. Likes humidity.',
      de: 'Erde feucht halten. Mag Luftfeuchtigkeit.',
      es: 'Mantener la tierra húmeda. Le gusta la humedad.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Bright indirect light for best color.',
      de: 'Helles indirektes Licht für beste Färbung.',
      es: 'Luz indirecta brillante para mejor color.',
    },
    toxicity: 'mildlyToxic',
  },
  'Peperomia obtusifolia': {
    commonName: { en: 'Baby Rubber Plant', de: 'Zwergpfeffer', es: 'Peperomia' },
    water: {
      en: 'Low to moderate — let soil dry between waterings.',
      de: 'Wenig bis mäßig — Erde zwischen dem Gießen trocknen lassen.',
      es: 'Poco a moderado — dejar secar entre riegos.',
    },
    frequency: {
      en: 'Every 10–14 days',
      de: 'Alle 10–14 Tage',
      es: 'Cada 10–14 días',
    },
    sunlight: {
      en: 'Medium to bright indirect light.',
      de: 'Mittleres bis helles indirektes Licht.',
      es: 'Luz indirecta media a brillante.',
    },
    toxicity: 'safe',
  },
  'Hoya carnosa': {
    commonName: { en: 'Wax Plant', de: 'Wachsblume', es: 'Flor de cera' },
    water: {
      en: 'Low — let soil dry out well between waterings.',
      de: 'Wenig — Erde gut trocknen lassen zwischen dem Gießen.',
      es: 'Poco — dejar secar bien entre riegos.',
    },
    frequency: {
      en: 'Every 10–14 days',
      de: 'Alle 10–14 Tage',
      es: 'Cada 10–14 días',
    },
    sunlight: {
      en: 'Bright indirect light. Some direct sun is okay.',
      de: 'Helles indirektes Licht. Etwas direkte Sonne ist in Ordnung.',
      es: 'Luz indirecta brillante. Algo de sol directo está bien.',
    },
    toxicity: 'safe',
  },
  'Aloe vera': {
    commonName: { en: 'Aloe Vera', de: 'Aloe Vera', es: 'Aloe vera' },
    water: {
      en: 'Sparingly — let soil dry completely.',
      de: 'Sparsam — Erde vollständig trocknen lassen.',
      es: 'Con moderación — dejar secar completamente.',
    },
    frequency: {
      en: 'Every 14–21 days',
      de: 'Alle 14–21 Tage',
      es: 'Cada 14–21 días',
    },
    sunlight: {
      en: 'Bright direct or indirect light.',
      de: 'Helles direktes oder indirektes Licht.',
      es: 'Luz directa o indirecta brillante.',
    },
    toxicity: 'mildlyToxic',
  },
  'Crassula ovata': {
    commonName: { en: 'Jade Plant', de: 'Geldbaum', es: 'Planta de jade' },
    water: {
      en: 'Sparingly — let soil dry out fully.',
      de: 'Sparsam — Erde vollständig trocknen lassen.',
      es: 'Con moderación — dejar secar completamente.',
    },
    frequency: {
      en: 'Every 14–21 days',
      de: 'Alle 14–21 Tage',
      es: 'Cada 14–21 días',
    },
    sunlight: {
      en: 'Bright light with some direct sun.',
      de: 'Helles Licht mit etwas direkter Sonne.',
      es: 'Luz brillante con algo de sol directo.',
    },
    toxicity: 'toxic',
  },
  'Dypsis lutescens': {
    commonName: { en: 'Areca Palm', de: 'Arecapalme', es: 'Palmera areca' },
    water: {
      en: 'Moderate — keep soil slightly moist.',
      de: 'Mäßig — Erde leicht feucht halten.',
      es: 'Moderado — mantener la tierra ligeramente húmeda.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light.',
      de: 'Helles indirektes Licht.',
      es: 'Luz indirecta brillante.',
    },
    toxicity: 'safe',
  },
  'Pachira aquatica': {
    commonName: { en: 'Money Tree', de: 'Glückskastanie', es: 'Árbol del dinero' },
    water: {
      en: 'Moderate — water when top soil is dry.',
      de: 'Mäßig — gießen, wenn die obere Erde trocken ist.',
      es: 'Moderado — regar cuando la superficie esté seca.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. Tolerates some shade.',
      de: 'Helles indirektes Licht. Verträgt etwas Schatten.',
      es: 'Luz indirecta brillante. Tolera algo de sombra.',
    },
    toxicity: 'safe',
  },
  'Yucca elephantipes': {
    commonName: { en: 'Spineless Yucca', de: 'Yucca-Palme', es: 'Yuca pie de elefante' },
    water: {
      en: 'Sparingly — let soil dry out between waterings.',
      de: 'Sparsam — Erde zwischen dem Gießen trocknen lassen.',
      es: 'Con moderación — dejar secar entre riegos.',
    },
    frequency: {
      en: 'Every 10–14 days',
      de: 'Alle 10–14 Tage',
      es: 'Cada 10–14 días',
    },
    sunlight: {
      en: 'Bright light. Tolerates direct sun.',
      de: 'Helles Licht. Verträgt direkte Sonne.',
      es: 'Luz brillante. Tolera sol directo.',
    },
    toxicity: 'mildlyToxic',
  },
  'Hedera helix': {
    commonName: { en: 'English Ivy', de: 'Efeu', es: 'Hiedra' },
    water: {
      en: 'Moderate — keep soil lightly moist.',
      de: 'Mäßig — Erde leicht feucht halten.',
      es: 'Moderado — mantener la tierra ligeramente húmeda.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Bright indirect light. Tolerates shade.',
      de: 'Helles indirektes Licht. Verträgt Schatten.',
      es: 'Luz indirecta brillante. Tolera la sombra.',
    },
    toxicity: 'toxic',
  },
  'Asplenium nidus': {
    commonName: { en: 'Bird\'s Nest Fern', de: 'Nestfarn', es: 'Helecho nido de ave' },
    water: {
      en: 'Keep soil consistently moist. Avoid watering the rosette center.',
      de: 'Erde gleichmäßig feucht halten. Nicht in die Rosettenmitte gießen.',
      es: 'Mantener la tierra húmeda. Evitar regar el centro de la roseta.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Low to medium indirect light. No direct sun.',
      de: 'Wenig bis mittleres indirektes Licht. Keine direkte Sonne.',
      es: 'Luz indirecta baja a media. Sin sol directo.',
    },
    toxicity: 'safe',
  },
  'Codiaeum variegatum': {
    commonName: { en: 'Croton', de: 'Kroton', es: 'Crotón' },
    water: {
      en: 'Moderate — keep soil evenly moist.',
      de: 'Mäßig — Erde gleichmäßig feucht halten.',
      es: 'Moderado — mantener la tierra uniformemente húmeda.',
    },
    frequency: {
      en: 'Every 5–7 days',
      de: 'Alle 5–7 Tage',
      es: 'Cada 5–7 días',
    },
    sunlight: {
      en: 'Bright direct or indirect light for best color.',
      de: 'Helles direktes oder indirektes Licht für beste Färbung.',
      es: 'Luz directa o indirecta brillante para mejor color.',
    },
    toxicity: 'toxic',
  },
  'Schefflera arboricola': {
    commonName: { en: 'Dwarf Umbrella Tree', de: 'Zwergschefflera', es: 'Árbol sombrilla enano' },
    water: {
      en: 'Moderate — let top soil dry between waterings.',
      de: 'Mäßig — obere Erde zwischen dem Gießen trocknen lassen.',
      es: 'Moderado — dejar secar la superficie entre riegos.',
    },
    frequency: {
      en: 'Every 7–10 days',
      de: 'Alle 7–10 Tage',
      es: 'Cada 7–10 días',
    },
    sunlight: {
      en: 'Bright indirect light. Some direct sun okay.',
      de: 'Helles indirektes Licht. Etwas direkte Sonne in Ordnung.',
      es: 'Luz indirecta brillante. Algo de sol directo está bien.',
    },
    toxicity: 'mildlyToxic',
  },
};

/**
 * Look up care info by scientific name.
 * Returns null if the plant isn't in the database.
 */
export function getPlantCare(scientificName) {
  return PLANT_CARE[scientificName] || null;
}

/**
 * Return a sorted list of all plants the model can identify.
 */
export function getPlantList() {
  return Object.entries(PLANT_CARE)
    .map(([scientific, info]) => ({ scientific, commonName: info.commonName }))
    .sort((a, b) => a.scientific.localeCompare(b.scientific));
}
