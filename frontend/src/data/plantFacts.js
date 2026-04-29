/**
 * Fun facts for species the model can identify.
 * Each entry has an array of { en, de, es } objects so we can pick one at random.
 */
const PLANT_FACTS = {
  'Monstera deliciosa': [
    {
      en: 'In the wild, Monstera can climb trees over 20 metres tall.',
      de: 'In der Natur kann die Monstera Bäume von über 20 Metern erklimmen.',
      es: 'En la naturaleza, la Monstera puede trepar árboles de más de 20 metros.',
    },
    {
      en: 'The holes in its leaves are called fenestrations and help it survive strong winds.',
      de: 'Die Löcher in den Blättern heißen Fenestrationen und helfen ihr, starken Wind zu überstehen.',
      es: 'Los agujeros en sus hojas se llaman fenestraciones y la ayudan a resistir vientos fuertes.',
    },
    {
      en: 'The fruit of the Monstera takes over a year to ripen and tastes like a mix of banana and pineapple.',
      de: 'Die Frucht der Monstera braucht über ein Jahr zum Reifen und schmeckt wie eine Mischung aus Banane und Ananas.',
      es: 'La fruta de la Monstera tarda más de un año en madurar y sabe a una mezcla de plátano y piña.',
    },
    {
      en: 'Young Monstera leaves are solid — the iconic holes only develop as the plant matures.',
      de: 'Junge Monstera-Blätter sind geschlossen — die ikonischen Löcher entstehen erst mit zunehmendem Alter.',
      es: 'Las hojas jóvenes de Monstera son enteras — los icónicos agujeros solo aparecen con la madurez.',
    },
  ],
  'Epipremnum aureum': [
    {
      en: 'Pothos is almost impossible to kill and can grow in just water.',
      de: 'Pothos ist fast unzerstörbar und kann sogar nur in Wasser wachsen.',
      es: 'El Pothos es casi imposible de matar y puede crecer solo en agua.',
    },
    {
      en: 'NASA listed Pothos as one of the best air-purifying houseplants.',
      de: 'Die NASA listete Pothos als eine der besten luftreinigenden Zimmerpflanzen.',
      es: 'La NASA incluyó al Pothos como una de las mejores plantas purificadoras de aire.',
    },
    {
      en: 'In the wild, Pothos leaves can grow up to 1 metre wide when climbing tall trees.',
      de: 'In der Natur können Pothos-Blätter bis zu 1 Meter breit werden, wenn sie an hohen Bäumen klettern.',
      es: 'En la naturaleza, las hojas del Pothos pueden crecer hasta 1 metro de ancho al trepar árboles altos.',
    },
    {
      en: 'Pothos changes leaf shape dramatically — heart-shaped indoors but deeply split when climbing in nature.',
      de: 'Pothos verändert seine Blattform stark — herzförmig drinnen, aber tief gespalten beim Klettern in der Natur.',
      es: 'El Pothos cambia drásticamente la forma de sus hojas — acorazonadas en interior pero profundamente divididas al trepar en la naturaleza.',
    },
  ],
  'Spathiphyllum wallisii': [
    {
      en: 'Peace Lilies aren\'t true lilies — they belong to the Araceae family.',
      de: 'Friedenslilien sind keine echten Lilien — sie gehören zur Familie der Araceae.',
      es: 'Los lirios de la paz no son lirios verdaderos — pertenecen a la familia Araceae.',
    },
    {
      en: 'They can bloom even in low-light rooms, which is rare for flowering plants.',
      de: 'Sie können sogar in dunklen Räumen blühen, was bei Blütenpflanzen selten ist.',
      es: 'Pueden florecer incluso en habitaciones con poca luz, algo raro en plantas con flores.',
    },
    {
      en: 'Peace Lilies will visibly droop when thirsty, but bounce back within hours of watering.',
      de: 'Friedenslilien lassen sichtbar die Blätter hängen, wenn sie durstig sind, erholen sich aber innerhalb von Stunden nach dem Gießen.',
      es: 'Los lirios de la paz se marchitan visiblemente cuando tienen sed, pero se recuperan en pocas horas tras el riego.',
    },
    {
      en: 'They are native to the tropical rainforests of Central and South America.',
      de: 'Sie stammen aus den tropischen Regenwäldern Mittel- und Südamerikas.',
      es: 'Son originarios de las selvas tropicales de América Central y del Sur.',
    },
  ],
  'Sansevieria trifasciata': [
    {
      en: 'Snake Plants release oxygen at night, making them great bedroom plants.',
      de: 'Bogenhanf gibt nachts Sauerstoff ab und ist daher ideal fürs Schlafzimmer.',
      es: 'Las lenguas de suegra liberan oxígeno por la noche, ideales para el dormitorio.',
    },
    {
      en: 'They can survive weeks without water thanks to succulent-like leaves.',
      de: 'Sie können dank sukkulentenartiger Blätter wochenlang ohne Wasser überleben.',
      es: 'Pueden sobrevivir semanas sin agua gracias a sus hojas suculentas.',
    },
    {
      en: 'Snake Plants use a special type of photosynthesis called CAM, which reduces water loss.',
      de: 'Bogenhanf nutzt eine spezielle Photosynthese namens CAM, die den Wasserverlust reduziert.',
      es: 'Las lenguas de suegra usan un tipo especial de fotosíntesis llamado CAM, que reduce la pérdida de agua.',
    },
    {
      en: 'They can be propagated by cutting a single leaf into sections and planting them.',
      de: 'Sie können vermehrt werden, indem man ein einzelnes Blatt in Abschnitte schneidet und einpflanzt.',
      es: 'Se pueden propagar cortando una sola hoja en secciones y plantándolas.',
    },
  ],
  'Chlorophytum comosum': [
    {
      en: 'Spider Plants produce "babies" on long runners that can be propagated easily.',
      de: 'Grünlilien bilden „Babys" an langen Ausläufern, die leicht vermehrt werden können.',
      es: 'Las cintas producen "bebés" en largos estolones que se propagan fácilmente.',
    },
    {
      en: 'They were among the first plants tested by NASA for air purification.',
      de: 'Sie gehörten zu den ersten Pflanzen, die von der NASA zur Luftreinigung getestet wurden.',
      es: 'Fueron de las primeras plantas probadas por la NASA para purificar el aire.',
    },
    {
      en: 'Spider Plants are non-toxic to cats and dogs, making them one of the safest houseplants.',
      de: 'Grünlilien sind ungiftig für Katzen und Hunde und damit eine der sichersten Zimmerpflanzen.',
      es: 'Las cintas no son tóxicas para perros y gatos, lo que las convierte en una de las plantas más seguras.',
    },
    {
      en: 'A single mature plant can produce dozens of babies in one growing season.',
      de: 'Eine einzige ausgewachsene Pflanze kann in einer Wachstumssaison Dutzende von Ablegern produzieren.',
      es: 'Una sola planta madura puede producir docenas de hijuelos en una temporada de crecimiento.',
    },
  ],
  'Ficus elastica': [
    {
      en: 'Rubber Plants were once tapped for latex used to make natural rubber.',
      de: 'Gummibäume wurden einst zur Gewinnung von Latex für Naturkautschuk angezapft.',
      es: 'Los árboles de caucho se usaban para extraer látex para hacer caucho natural.',
    },
    {
      en: 'In tropical forests, their aerial roots can strangle host trees.',
      de: 'In tropischen Wäldern können ihre Luftwurzeln Wirtsbäume erwürgen.',
      es: 'En bosques tropicales, sus raíces aéreas pueden estrangular árboles huéspedes.',
    },
    {
      en: 'New Rubber Plant leaves emerge from a bright red sheath that falls off as the leaf unfurls.',
      de: 'Neue Gummibaum-Blätter erscheinen aus einer leuchtend roten Hülle, die abfällt, wenn sich das Blatt entfaltet.',
      es: 'Las hojas nuevas del árbol de caucho emergen de una vaina roja brillante que cae al desplegarse la hoja.',
    },
    {
      en: 'In India, living root bridges are built from related Ficus species and can last for centuries.',
      de: 'In Indien werden aus verwandten Ficus-Arten lebende Wurzelbrücken gebaut, die Jahrhunderte halten können.',
      es: 'En India se construyen puentes de raíces vivas con especies de Ficus emparentadas que pueden durar siglos.',
    },
  ],
  'Ficus benjamina': [
    {
      en: 'Weeping Figs famously drop their leaves when moved to a new spot.',
      de: 'Birkenfeigen verlieren bekanntlich ihre Blätter, wenn sie umgestellt werden.',
      es: 'Los ficus benjamina pierden hojas cuando se cambian de lugar.',
    },
    {
      en: 'In warm climates they can grow into large trees exceeding 30 metres.',
      de: 'In warmen Klimazonen können sie zu über 30 Meter hohen Bäumen heranwachsen.',
      es: 'En climas cálidos pueden crecer como árboles de más de 30 metros.',
    },
    {
      en: 'Weeping Figs are the official tree of Bangkok, Thailand.',
      de: 'Die Birkenfeige ist der offizielle Baum von Bangkok, Thailand.',
      es: 'El ficus benjamina es el árbol oficial de Bangkok, Tailandia.',
    },
    {
      en: 'They can form aerial roots that thicken into additional trunks over time.',
      de: 'Sie können Luftwurzeln bilden, die sich mit der Zeit zu zusätzlichen Stämmen verdicken.',
      es: 'Pueden formar raíces aéreas que se engrosan y se convierten en troncos adicionales con el tiempo.',
    },
  ],
  'Dieffenbachia seguine': [
    {
      en: 'Chewing a Dieffenbachia leaf can temporarily make you unable to speak — hence "Dumb Cane".',
      de: 'Das Kauen eines Dieffenbachia-Blatts kann vorübergehend die Sprache rauben — daher „Dumb Cane".',
      es: 'Masticar una hoja de Dieffenbachia puede dejarte temporalmente sin habla — de ahí "Dumb Cane".',
    },
    {
      en: 'It was one of the first tropical plants brought to Europe in the 1800s.',
      de: 'Sie war eine der ersten tropischen Pflanzen, die im 19. Jahrhundert nach Europa gebracht wurden.',
      es: 'Fue una de las primeras plantas tropicales traídas a Europa en el siglo XIX.',
    },
    {
      en: 'The toxicity is caused by needle-like calcium oxalate crystals that pierce mouth tissue.',
      de: 'Die Giftigkeit wird durch nadelförmige Calciumoxalat-Kristalle verursacht, die das Mundgewebe durchstechen.',
      es: 'La toxicidad es causada por cristales de oxalato de calcio en forma de aguja que perforan el tejido bucal.',
    },
    {
      en: 'Dieffenbachia can grow over 1.5 metres tall indoors with the right care.',
      de: 'Dieffenbachia kann bei richtiger Pflege in Innenräumen über 1,5 Meter hoch werden.',
      es: 'La Dieffenbachia puede crecer más de 1,5 metros en interiores con el cuidado adecuado.',
    },
  ],
  'Alocasia amazonica': [
    {
      en: 'In some cultures, giant Alocasia leaves are used as natural umbrellas.',
      de: 'In manchen Kulturen werden riesige Alocasia-Blätter als natürliche Regenschirme verwendet.',
      es: 'En algunas culturas, las hojas gigantes de Alocasia se usan como paraguas naturales.',
    },
    {
      en: 'The species name "amazonica" is misleading — it was bred in a Miami nursery.',
      de: 'Der Artname „amazonica" ist irreführend — sie wurde in einer Gärtnerei in Miami gezüchtet.',
      es: 'El nombre "amazonica" es engañoso — fue criada en un vivero de Miami.',
    },
    {
      en: 'Alocasia leaves can "drip" water from their tips through a process called guttation.',
      de: 'Alocasia-Blätter können durch einen Prozess namens Guttation Wasser von ihren Spitzen „tropfen".',
      es: 'Las hojas de Alocasia pueden "gotear" agua desde sus puntas mediante un proceso llamado gutación.',
    },
    {
      en: 'They go dormant in winter and may lose all their leaves before regrowing in spring.',
      de: 'Sie gehen im Winter in die Ruhephase und können alle Blätter verlieren, bevor sie im Frühling neu austreiben.',
      es: 'Entran en latencia en invierno y pueden perder todas sus hojas antes de rebrotar en primavera.',
    },
  ],
  'Anthurium andraeanum': [
    {
      en: 'The colorful "petal" of an Anthurium is actually a modified leaf called a spathe.',
      de: 'Das farbige „Blütenblatt" eines Anthuriums ist eigentlich ein modifiziertes Blatt, die Spatha.',
      es: 'El "pétalo" colorido del Anturio en realidad es una hoja modificada llamada espata.',
    },
    {
      en: 'They can bloom continuously for months under good conditions.',
      de: 'Unter guten Bedingungen können sie monatelang durchgehend blühen.',
      es: 'Pueden florecer continuamente durante meses en buenas condiciones.',
    },
    {
      en: 'Anthuriums are among the longest-lasting cut flowers, staying fresh for up to 8 weeks in a vase.',
      de: 'Anthurien gehören zu den langlebigsten Schnittblumen und bleiben bis zu 8 Wochen frisch in der Vase.',
      es: 'Los anturios son de las flores cortadas más duraderas, manteniéndose frescas hasta 8 semanas en un jarrón.',
    },
    {
      en: 'They are native to the cloud forests of Colombia and Ecuador.',
      de: 'Sie stammen aus den Nebelwäldern Kolumbiens und Ecuadors.',
      es: 'Son originarios de los bosques nubosos de Colombia y Ecuador.',
    },
  ],
  'Philodendron hederaceum': [
    {
      en: 'Heartleaf Philodendrons have been popular houseplants since the Victorian era.',
      de: 'Herzblatt-Philodendren sind seit der viktorianischen Ära beliebte Zimmerpflanzen.',
      es: 'Los filodendros de hoja corazón son plantas de interior populares desde la era victoriana.',
    },
    {
      en: 'They are epiphytes — in the wild they climb other trees for light.',
      de: 'Sie sind Epiphyten — in der Natur klettern sie an Bäumen empor zum Licht.',
      es: 'Son epífitas — en la naturaleza trepan otros árboles buscando luz.',
    },
    {
      en: 'The name Philodendron means "tree lover" in Greek, referring to its climbing habit.',
      de: 'Der Name Philodendron bedeutet auf Griechisch „Baumliebhaber" und bezieht sich auf seine Klettergewohnheit.',
      es: 'El nombre Philodendron significa "amante de los árboles" en griego, refiriéndose a su hábito trepador.',
    },
    {
      en: 'Heartleaf Philodendrons can trail over 3 metres long when grown in hanging baskets.',
      de: 'Herzblatt-Philodendren können in Hängeampeln über 3 Meter lang herabhängen.',
      es: 'Los filodendros de hoja corazón pueden colgar más de 3 metros en macetas colgantes.',
    },
  ],
  'Tradescantia zebrina': [
    {
      en: 'Tradescantia can root from a single stem cutting placed in water.',
      de: 'Tradescantia kann aus einem einzelnen Steckling im Wasser Wurzeln schlagen.',
      es: 'La Tradescantia puede enraizar con un solo esqueje en agua.',
    },
    {
      en: 'Its purple-striped leaves shimmer with a metallic sheen in sunlight.',
      de: 'Ihre lila gestreiften Blätter schimmern im Sonnenlicht metallisch.',
      es: 'Sus hojas de rayas púrpura brillan con un destello metálico al sol.',
    },
    {
      en: 'Tradescantia is named after John Tradescant, gardener to King Charles I of England.',
      de: 'Tradescantia ist nach John Tradescant benannt, dem Gärtner von König Karl I. von England.',
      es: 'La Tradescantia lleva el nombre de John Tradescant, jardinero del rey Carlos I de Inglaterra.',
    },
    {
      en: 'It grows so fast that it is considered invasive in tropical regions around the world.',
      de: 'Sie wächst so schnell, dass sie in tropischen Regionen weltweit als invasiv gilt.',
      es: 'Crece tan rápido que se considera invasora en regiones tropicales de todo el mundo.',
    },
  ],
  'Peperomia obtusifolia': [
    {
      en: 'There are over 1,500 species of Peperomia worldwide.',
      de: 'Es gibt weltweit über 1.500 Peperomia-Arten.',
      es: 'Hay más de 1.500 especies de Peperomia en el mundo.',
    },
    {
      en: 'Their thick leaves store water, making them semi-succulent.',
      de: 'Ihre dicken Blätter speichern Wasser, was sie zu Halbsukkulenten macht.',
      es: 'Sus hojas gruesas almacenan agua, haciéndolas semi-suculentas.',
    },
    {
      en: 'Peperomias are related to black pepper plants and belong to the family Piperaceae.',
      de: 'Peperomien sind mit dem schwarzen Pfeffer verwandt und gehören zur Familie der Piperaceae.',
      es: 'Las Peperomias están emparentadas con la pimienta negra y pertenecen a la familia Piperaceae.',
    },
    {
      en: 'They produce unusual flower spikes that look like tiny green rat tails.',
      de: 'Sie bilden ungewöhnliche Blütenähren, die wie kleine grüne Rattenschwänze aussehen.',
      es: 'Producen espigas florales inusuales que parecen pequeñas colas de rata verdes.',
    },
  ],
  'Hoya carnosa': [
    {
      en: 'Hoya flowers produce fragrant nectar droplets that are sticky to the touch.',
      de: 'Hoya-Blüten produzieren duftende Nektartropfen, die klebrig sind.',
      es: 'Las flores de Hoya producen gotas de néctar fragante que son pegajosas al tacto.',
    },
    {
      en: 'A single Hoya vine can live for decades and grow several metres long.',
      de: 'Eine einzelne Hoya-Ranke kann jahrzehntelang leben und mehrere Meter lang werden.',
      es: 'Una sola enredadera de Hoya puede vivir décadas y crecer varios metros.',
    },
    {
      en: 'Hoyas are also called "wax plants" because their leaves look as if dipped in wax.',
      de: 'Hoyas werden auch „Wachsblumen" genannt, weil ihre Blätter wie in Wachs getaucht aussehen.',
      es: 'Las Hoyas también se llaman "plantas de cera" porque sus hojas parecen sumergidas en cera.',
    },
    {
      en: 'Removing old flower spurs is a mistake — Hoyas rebloom from the same spur year after year.',
      de: 'Alte Blütenstiele zu entfernen ist ein Fehler — Hoyas blühen Jahr für Jahr am selben Stiel erneut.',
      es: 'Quitar los pedúnculos viejos es un error — las Hoyas vuelven a florecer del mismo pedúnculo año tras año.',
    },
  ],
  'Aloe vera': [
    {
      en: 'Aloe vera gel has been used medicinally for over 6,000 years.',
      de: 'Aloe-Vera-Gel wird seit über 6.000 Jahren medizinisch verwendet.',
      es: 'El gel de aloe vera se ha usado con fines medicinales durante más de 6.000 años.',
    },
    {
      en: 'Ancient Egyptians called Aloe the "plant of immortality".',
      de: 'Die alten Ägypter nannten Aloe die „Pflanze der Unsterblichkeit".',
      es: 'Los antiguos egipcios llamaban al Aloe la "planta de la inmortalidad".',
    },
    {
      en: 'Aloe leaves can hold up to 99% water, which helps them survive in arid deserts.',
      de: 'Aloe-Blätter können bis zu 99 % Wasser speichern, was ihnen hilft, in trockenen Wüsten zu überleben.',
      es: 'Las hojas de Aloe pueden contener hasta un 99 % de agua, lo que les ayuda a sobrevivir en desiertos áridos.',
    },
    {
      en: 'Cleopatra reportedly used Aloe vera gel as part of her daily beauty routine.',
      de: 'Kleopatra soll Aloe-Vera-Gel als Teil ihrer täglichen Schönheitspflege verwendet haben.',
      es: 'Se dice que Cleopatra usaba gel de aloe vera como parte de su rutina diaria de belleza.',
    },
  ],
  'Crassula ovata': [
    {
      en: 'In many cultures, Jade Plants are considered symbols of good luck and prosperity.',
      de: 'In vielen Kulturen gelten Geldbaum-Pflanzen als Symbole für Glück und Wohlstand.',
      es: 'En muchas culturas, las plantas de jade se consideran símbolos de buena suerte y prosperidad.',
    },
    {
      en: 'They can live for 50+ years and develop thick, tree-like trunks.',
      de: 'Sie können über 50 Jahre alt werden und dicke, baumartige Stämme entwickeln.',
      es: 'Pueden vivir más de 50 años y desarrollar troncos gruesos como árboles.',
    },
    {
      en: 'Jade Plants can be propagated from a single fallen leaf that roots in soil.',
      de: 'Geldbaum-Pflanzen können aus einem einzigen abgefallenen Blatt vermehrt werden, das in Erde wurzelt.',
      es: 'Las plantas de jade se pueden propagar a partir de una sola hoja caída que enraíza en la tierra.',
    },
    {
      en: 'Under stress from cold or sun, their leaf edges turn a striking red colour.',
      de: 'Unter Stress durch Kälte oder Sonne färben sich ihre Blattränder auffällig rot.',
      es: 'Bajo estrés por frío o sol, los bordes de sus hojas se tornan de un llamativo color rojo.',
    },
  ],
  'Dypsis lutescens': [
    {
      en: 'Areca Palms can transpire up to a litre of water per day, humidifying a room.',
      de: 'Arecapalmen können bis zu einen Liter Wasser pro Tag verdunsten und so einen Raum befeuchten.',
      es: 'Las palmeras areca pueden transpirar hasta un litro de agua al día, humidificando una habitación.',
    },
    {
      en: 'In the wild they grow in clusters and can reach 12 metres tall.',
      de: 'In der Natur wachsen sie in Gruppen und können 12 Meter hoch werden.',
      es: 'En la naturaleza crecen en grupos y pueden alcanzar los 12 metros.',
    },
    {
      en: 'Areca Palms are listed as endangered in their native Madagascar habitat.',
      de: 'Arecapalmen gelten in ihrem Heimathabitat Madagaskar als gefährdet.',
      es: 'Las palmeras areca están clasificadas como en peligro en su hábitat nativo de Madagascar.',
    },
    {
      en: 'Their seeds are used as a betel nut substitute in some parts of India.',
      de: 'Ihre Samen werden in einigen Teilen Indiens als Betelnuss-Ersatz verwendet.',
      es: 'Sus semillas se usan como sustituto de la nuez de betel en algunas partes de India.',
    },
  ],
  'Pachira aquatica': [
    {
      en: 'Money Trees are often sold with braided trunks — this is done by hand while young.',
      de: 'Geldbaum-Stämme werden oft geflochten — das geschieht von Hand, solange sie jung sind.',
      es: 'Los árboles del dinero se venden a menudo con troncos trenzados a mano mientras son jóvenes.',
    },
    {
      en: 'In the wild, Pachira can grow up to 18 metres and produces large edible nuts.',
      de: 'In der Natur kann die Pachira bis zu 18 Meter hoch werden und große essbare Nüsse tragen.',
      es: 'En la naturaleza, la Pachira puede crecer hasta 18 metros y produce grandes nueces comestibles.',
    },
    {
      en: 'The "money tree" legend originated in Taiwan in the 1980s when a farmer grew braided specimens.',
      de: 'Die „Geldbaum"-Legende entstand in den 1980ern in Taiwan, als ein Bauer geflochtene Exemplare züchtete.',
      es: 'La leyenda del "árbol del dinero" se originó en Taiwán en los años 80 cuando un granjero cultivó ejemplares trenzados.',
    },
    {
      en: 'Pachira naturally grows along rivers and can tolerate periodic flooding.',
      de: 'Pachira wächst natürlich entlang von Flüssen und verträgt periodische Überschwemmungen.',
      es: 'La Pachira crece naturalmente junto a ríos y puede tolerar inundaciones periódicas.',
    },
  ],
  'Yucca elephantipes': [
    {
      en: 'Yucca flowers are edible and considered a delicacy in Central American cuisine.',
      de: 'Yucca-Blüten sind essbar und gelten in der mittelamerikanischen Küche als Delikatesse.',
      es: 'Las flores de yuca son comestibles y se consideran un manjar en la cocina centroamericana.',
    },
    {
      en: 'Despite their tropical look, Yuccas are incredibly drought-tolerant desert plants.',
      de: 'Trotz ihres tropischen Aussehens sind Yuccas unglaublich trockenheitsresistente Wüstenpflanzen.',
      es: 'A pesar de su aspecto tropical, las yucas son plantas desérticas increíblemente resistentes a la sequía.',
    },
    {
      en: 'Yuccas are pollinated exclusively by yucca moths in a classic example of mutualism.',
      de: 'Yuccas werden ausschließlich von Yucca-Motten bestäubt — ein klassisches Beispiel für Mutualismus.',
      es: 'Las yucas son polinizadas exclusivamente por polillas de yuca, un ejemplo clásico de mutualismo.',
    },
    {
      en: 'The trunk stores water, which is why it develops a swollen, elephant-foot-like base.',
      de: 'Der Stamm speichert Wasser, weshalb er eine geschwollene, elefantenfußartige Basis entwickelt.',
      es: 'El tronco almacena agua, por eso desarrolla una base hinchada similar a una pata de elefante.',
    },
  ],
  'Hedera helix': [
    {
      en: 'English Ivy can live for centuries — some specimens in Europe are over 400 years old.',
      de: 'Efeu kann Jahrhunderte alt werden — einige Exemplare in Europa sind über 400 Jahre alt.',
      es: 'La hiedra puede vivir siglos — algunos ejemplares en Europa tienen más de 400 años.',
    },
    {
      en: 'Ivy produces two types of leaves: lobed juvenile leaves and oval adult leaves.',
      de: 'Efeu bildet zwei Blattformen: gelappte Jugendblätter und ovale Altersblätter.',
      es: 'La hiedra produce dos tipos de hojas: lobuladas de juvenil y ovaladas de adulta.',
    },
    {
      en: 'Ivy was shown to reduce indoor mould spores by up to 78% in studies.',
      de: 'Studien zeigten, dass Efeu Schimmelsporen in Innenräumen um bis zu 78 % reduzieren kann.',
      es: 'Estudios demostraron que la hiedra puede reducir las esporas de moho en interiores hasta un 78 %.',
    },
    {
      en: 'Ancient Greeks associated ivy with the god Dionysus and used it as a symbol of fidelity.',
      de: 'Die alten Griechen verbanden Efeu mit dem Gott Dionysos und nutzten ihn als Symbol der Treue.',
      es: 'Los antiguos griegos asociaban la hiedra con el dios Dioniso y la usaban como símbolo de fidelidad.',
    },
  ],
  'Asplenium nidus': [
    {
      en: 'Bird\'s Nest Ferns collect rainwater and debris in their rosette — a natural compost pile.',
      de: 'Nestfarne sammeln Regenwasser und Laub in ihrer Rosette — ein natürlicher Komposthaufen.',
      es: 'Los helechos nido de ave recolectan agua de lluvia y residuos en su roseta — un compost natural.',
    },
    {
      en: 'They are epiphytes, growing on tree trunks in tropical rainforests without soil.',
      de: 'Sie sind Epiphyten und wachsen in tropischen Regenwäldern ohne Erde auf Baumstämmen.',
      es: 'Son epífitas, crecen sobre troncos en selvas tropicales sin necesitar suelo.',
    },
    {
      en: 'New fronds unfurl from the centre of the rosette in a tight spiral called a fiddlehead.',
      de: 'Neue Wedel entfalten sich aus der Mitte der Rosette in einer engen Spirale, dem sogenannten Fiddlehead.',
      es: 'Las nuevas frondas se despliegan desde el centro de la roseta en una espiral apretada llamada cabeza de violín.',
    },
    {
      en: 'In Southeast Asia, young fiddleheads of certain Asplenium species are eaten as a vegetable.',
      de: 'In Südostasien werden junge Fiddleheads bestimmter Asplenium-Arten als Gemüse gegessen.',
      es: 'En el sudeste asiático, los brotes jóvenes de ciertas especies de Asplenium se comen como verdura.',
    },
  ],
  'Codiaeum variegatum': [
    {
      en: 'Crotons can display red, orange, yellow, green, and purple — all on the same plant.',
      de: 'Kroton kann Rot, Orange, Gelb, Grün und Lila zeigen — alles an derselben Pflanze.',
      es: 'Los crotones pueden mostrar rojo, naranja, amarillo, verde y púrpura — todo en la misma planta.',
    },
    {
      en: 'Their vivid colours come from anthocyanins and carotenoids reacting to bright light.',
      de: 'Ihre leuchtenden Farben entstehen durch Anthocyane und Carotinoide, die auf helles Licht reagieren.',
      es: 'Sus vívidos colores provienen de antocianinas y carotenoides que reaccionan a la luz intensa.',
    },
    {
      en: 'Crotons need very bright light — their colours fade to plain green in low light conditions.',
      de: 'Kroton braucht sehr helles Licht — seine Farben verblassen bei wenig Licht zu schlichtem Grün.',
      es: 'Los crotones necesitan mucha luz — sus colores se desvanecen a verde en condiciones de poca luz.',
    },
    {
      en: 'They are native to Indonesia and Malaysia, where they grow as shrubs up to 3 metres tall.',
      de: 'Sie stammen aus Indonesien und Malaysia, wo sie als Sträucher bis zu 3 Meter hoch wachsen.',
      es: 'Son originarios de Indonesia y Malasia, donde crecen como arbustos de hasta 3 metros de alto.',
    },
  ],
  'Schefflera arboricola': [
    {
      en: 'Dwarf Umbrella Trees are popular bonsai subjects due to their forgiving nature.',
      de: 'Zwergscheffleren sind wegen ihrer Pflegeleichtigkeit beliebte Bonsai-Pflanzen.',
      es: 'Los árboles sombrilla enanos son populares como bonsái por su naturaleza tolerante.',
    },
    {
      en: 'They get their name from the umbrella-like arrangement of their leaflets.',
      de: 'Ihren Namen verdanken sie der schirmartigen Anordnung ihrer Blättchen.',
      es: 'Deben su nombre a la disposición de sus folíolos en forma de paraguas.',
    },
    {
      en: 'Schefflera leaves fold up at night in a movement known as nyctinasty.',
      de: 'Schefflera-Blätter falten sich nachts zusammen — eine Bewegung namens Nyktinastie.',
      es: 'Las hojas de Schefflera se pliegan por la noche en un movimiento conocido como nictinastia.',
    },
    {
      en: 'In its native Taiwan, it can grow as an epiphyte on other trees in humid forests.',
      de: 'In ihrem Heimatland Taiwan kann sie als Epiphyt auf anderen Bäumen in feuchten Wäldern wachsen.',
      es: 'En su Taiwán natal, puede crecer como epífita sobre otros árboles en bosques húmedos.',
    },
  ],
};

/**
 * Return a random fun fact for a given species, or null if none available.
 * Returns the multilingual object { en, de, es }.
 */
export function getRandomFact(scientificName) {
  const facts = PLANT_FACTS[scientificName];
  if (!facts || facts.length === 0) return null;
  return facts[Math.floor(Math.random() * facts.length)];
}

/* Track the current index per species so we cycle without repeats. */
const _factIndex = {};

/**
 * Return the next fact for a species, cycling through all facts in order.
 * Guarantees no repeat until every fact has been shown.
 */
export function getNextFact(scientificName) {
  const facts = PLANT_FACTS[scientificName];
  if (!facts || facts.length === 0) return null;
  const prev = _factIndex[scientificName] ?? -1;
  const next = (prev + 1) % facts.length;
  _factIndex[scientificName] = next;
  return facts[next];
}
