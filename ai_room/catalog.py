"""The extended subject catalogue (90 subjects on top of the 10 core ones).

Each line is `key | Name | question types | six milestone topics`, ordered
from basic to advanced. The six milestones are spread over the 12 classes:
odd classes introduce a milestone, even classes apply it and connect it to
what came before (see `expand_syllabus`).
"""

from __future__ import annotations

Q = "numeric,mcq,cloze"   # quantitative subjects
C = "code,mcq,cloze"      # programming subjects
T = "mcq,cloze"           # conceptual / written subjects

CATALOG = f"""
# Mathematics & Statistics
statistics | Statistics | {Q} | descriptive statistics and data displays; sampling and the normal distribution; confidence intervals and hypothesis tests; regression and correlation; ANOVA and nonparametric tests; Bayesian and multivariate statistics
probability | Probability | {Q} | counting and basic probability; conditional probability and Bayes' rule; discrete random variables and expectation; continuous distributions; limit theorems (LLN, CLT); Markov chains and stochastic processes
geometry | Geometry | {Q} | points, lines, angles and polygons; congruence, similarity and circles; coordinate geometry and transformations; solid geometry and volumes; non-Euclidean geometry; differential geometry of curves and surfaces
linear_algebra | Linear Algebra | {Q} | vectors and vector operations; matrices and linear systems; determinants and inverses; vector spaces and linear maps; eigenvalues and diagonalisation; SVD, inner product spaces and applications
calculus | Calculus | {Q} | rates of change and limits; differentiation rules; applications of derivatives; integration and the fundamental theorem; series and Taylor expansions; multivariable and vector calculus
number_theory | Number Theory | {Q} | divisibility and primes; GCD, LCM and the Euclidean algorithm; modular arithmetic; congruences and Fermat/Euler theorems; quadratic residues and Diophantine equations; analytic and algebraic number theory ideas
discrete_math | Discrete Mathematics | {Q} | sets, relations and functions; combinatorics and the pigeonhole principle; induction and recursion; graph theory basics; recurrence relations and generating functions; Ramsey theory and extremal combinatorics
optimization | Optimization | {Q} | maximising and minimising simple functions; linear programming; convex sets and functions; gradient methods; constrained optimization and Lagrange multipliers; integer, stochastic and nonconvex optimization
numerical_methods | Numerical Methods | {Q} | rounding and floating-point error; root finding (bisection, Newton); interpolation and numerical differentiation; numerical integration; solving linear systems and ODEs numerically; stability, convergence and PDE solvers

# Physical Sciences
astronomy | Astronomy | {Q} | the sky, day/night and seasons; the solar system; stars and their life cycles; galaxies and the Milky Way; telescopes and observational methods; exoplanets and astrobiology
astrophysics | Astrophysics | {Q} | light and the electromagnetic spectrum; stellar structure and fusion; compact objects (white dwarfs, neutron stars, black holes); galaxy dynamics and dark matter; cosmology and the Big Bang; gravitational waves and high-energy astrophysics
earth_science | Earth Science | {T} | Earth's layers and rocks; plate tectonics and earthquakes; the rock and water cycles; weathering, erosion and landforms; Earth's history and fossils; geophysics and planetary science
meteorology | Meteorology | {Q} | weather vs climate; the atmosphere's layers and composition; pressure, wind and fronts; clouds, precipitation and storms; forecasting and weather models; climate dynamics and climate change
oceanography | Oceanography | {T} | the world ocean and seawater; tides and waves; ocean currents and circulation; marine ecosystems; ocean-atmosphere interaction; ocean chemistry, acidification and research methods
materials_science | Materials Science | {Q} | types of materials; atomic structure and crystal lattices; mechanical properties and testing; phase diagrams; electronic and magnetic materials; nanomaterials and composites
electronics | Electronics | {Q} | voltage, current and resistance; resistors, capacitors and inductors; diodes and transistors; amplifiers and op-amps; digital logic circuits; microcontrollers and embedded systems
quantum_computing | Quantum Computing | {Q} | bits vs qubits; superposition and measurement; quantum gates and circuits; entanglement and teleportation; quantum algorithms (Deutsch-Jozsa, Grover, Shor); error correction and hardware platforms
organic_chemistry | Organic Chemistry | {T} | carbon bonding and hydrocarbons; functional groups and nomenclature; isomerism and stereochemistry; substitution and elimination reactions; carbonyl chemistry and aromaticity; multi-step synthesis and spectroscopy
biochemistry | Biochemistry | {T} | biomolecules overview; proteins and amino acids; enzymes and kinetics; carbohydrate and lipid metabolism; DNA, RNA and protein synthesis; signalling and metabolic regulation

# Life & Health Sciences
genetics | Genetics | {T} | traits and heredity; Mendelian genetics; DNA structure and replication; mutations and genetic disorders; population and quantitative genetics; genomics and gene editing
ecology | Ecology | {T} | organisms and habitats; food webs and energy flow; populations and growth; communities and species interactions; ecosystems and nutrient cycles; conservation biology and global change
neuroscience | Neuroscience | {T} | the nervous system overview; neurons and action potentials; synapses and neurotransmitters; sensory and motor systems; learning, memory and plasticity; cognitive and computational neuroscience
microbiology | Microbiology | {T} | microbes and the microscope; bacterial structure and growth; viruses and their replication; microbial metabolism and genetics; immunity and infection; microbiomes and biotechnology
botany | Botany | {T} | plant parts and functions; photosynthesis; plant reproduction; plant growth and hormones; plant diversity and evolution; plant physiology and crop science
zoology | Zoology | {T} | animal classification; invertebrates; vertebrates; animal behaviour; animal physiology; evolution and biogeography of animals
anatomy | Human Anatomy | {T} | body organisation and terminology; skeletal and muscular systems; cardiovascular and respiratory systems; digestive and urinary systems; nervous and endocrine systems; integrated clinical anatomy
medicine | Medicine | {T} | health, disease and the body; history taking and vital signs; pathophysiology of common diseases; diagnosis and diagnostic tests; treatment principles and evidence-based medicine; clinical reasoning with complex cases
nutrition | Nutrition | {T} | food groups and a balanced diet; macronutrients; vitamins and minerals; energy balance and metabolism; nutrition across the lifespan; clinical and public-health nutrition
psychology | Psychology | {T} | what psychology studies; perception and attention; learning and memory; development and personality; social psychology; psychological disorders and research methods
public_health | Public Health | {T} | health and communities; hygiene and disease prevention; epidemiology basics; health policy and systems; global health; outbreak investigation and biostatistics
pharmacology | Pharmacology | {T} | what drugs are; pharmacokinetics (ADME); pharmacodynamics and receptors; drug classes by system; drug interactions and toxicity; drug discovery and clinical trials

# Engineering
mechanical_eng | Mechanical Engineering | {Q} | forces and simple mechanisms; statics and free-body diagrams; strength of materials; dynamics and vibrations; thermodynamics and heat transfer; machine design and fluid mechanics
electrical_eng | Electrical Engineering | {Q} | circuits and Ohm's law; circuit analysis (KVL, KCL, Thevenin); AC power and phasors; electromagnetics; power systems and machines; power electronics and smart grids
civil_eng | Civil Engineering | {Q} | structures around us; loads and structural analysis; concrete and steel design; geotechnical engineering; transportation and water resources engineering; earthquake and sustainable design
chemical_eng | Chemical Engineering | {Q} | mass balances; energy balances; fluid flow and heat exchangers; mass transfer and separations; reaction engineering; process control and design
aerospace_eng | Aerospace Engineering | {Q} | how things fly; aerodynamics and lift; aircraft performance and stability; propulsion; orbital mechanics; spacecraft design
robotics | Robotics | {C} | what robots are; sensors and actuators; kinematics; motion planning; robot control and localisation (SLAM); learning-based and multi-robot systems
control_systems | Control Systems | {Q} | feedback and open vs closed loop; transfer functions and block diagrams; stability and the root locus; frequency response (Bode, Nyquist); PID and state-space control; optimal and robust control
signal_processing | Signal Processing | {Q} | signals and sampling; convolution and LTI systems; Fourier series and transforms; filters and filter design; the z-transform and DSP; spectral estimation and adaptive filters
environmental_eng | Environmental Engineering | {T} | pollution and the environment; water treatment; air quality engineering; solid and hazardous waste; environmental impact assessment; sustainable and climate-resilient design
biomedical_eng | Biomedical Engineering | {T} | engineering meets medicine; biomechanics; biomaterials; medical imaging; medical devices and instrumentation; tissue engineering and regulatory science

# Computing & AI
data_structures | Data Structures | {C} | arrays and strings; linked lists, stacks and queues; hash tables; trees and binary search trees; heaps and priority queues; graphs, tries and advanced structures
algorithms | Algorithms | {C} | step-by-step procedures and pseudocode; searching and sorting; divide and conquer; greedy algorithms; dynamic programming; graph algorithms and NP-completeness
machine_learning | Machine Learning | {C} | what learning from data means; linear and logistic regression; overfitting, validation and regularisation; trees, ensembles and SVMs; unsupervised learning and clustering; probabilistic models and learning theory
deep_learning | Deep Learning | {C} | neurons and perceptrons; backpropagation and gradient descent; convolutional networks; recurrent networks and sequence models; attention and transformers; scaling laws, generative models and interpretability
nlp | Natural Language Processing | {C} | text as data and tokenisation; n-grams and language models; word embeddings; sequence labelling and parsing; transformers and pretrained LLMs; retrieval, alignment and evaluation of LLMs
computer_vision | Computer Vision | {C} | images as arrays of pixels; filtering and edge detection; features and image matching; CNN image classification; object detection and segmentation; 3D vision and vision-language models
databases | Databases | {C} | tables, rows and keys; SQL queries; joins and aggregation; normalisation and schema design; indexes, transactions and concurrency; distributed and NoSQL databases
operating_systems | Operating Systems | {T} | what an OS does; processes and threads; CPU scheduling; memory management and virtual memory; file systems and I/O; synchronisation, deadlocks and virtualisation
networking | Computer Networks | {T} | what a network is; the OSI and TCP/IP models; IP addressing and routing; TCP, UDP and congestion control; DNS, HTTP and application protocols; network security and software-defined networking
cybersecurity | Cybersecurity | {T} | safe computing habits; the CIA triad and threat models; authentication and access control; cryptography in practice; web and network vulnerabilities (defensive view); incident response, secure design and auditing
web_dev | Web Development | {C} | how the web works; HTML and CSS; JavaScript and the DOM; HTTP APIs and backends; frameworks, state and databases; performance, accessibility and security
software_engineering | Software Engineering | {C} | the software lifecycle; version control and code review; testing and debugging; design principles and patterns; architecture and refactoring; reliability, scaling and project management
distributed_systems | Distributed Systems | {T} | why systems are distributed; communication and RPC; time, clocks and ordering; replication and consistency models; consensus (Paxos, Raft); fault tolerance and large-scale system design
compilers | Compilers | {C} | from source code to machine code; lexical analysis; parsing and grammars; semantic analysis and type checking; intermediate representations and code generation; optimisation and JIT compilation
data_science | Data Science | {C} | asking questions with data; data cleaning with Python; exploratory analysis and visualisation; feature engineering and modelling; experimentation and A/B testing; causal inference and communicating results
cloud_computing | Cloud Computing | {T} | what the cloud is; virtual machines and storage; containers and orchestration; serverless and managed services; scalability, cost and reliability; cloud security and multi-region architecture
reinforcement_learning | Reinforcement Learning | {C} | agents, rewards and environments; multi-armed bandits; Markov decision processes and dynamic programming; Monte Carlo and temporal-difference learning; deep RL and policy gradients; model-based, offline and multi-agent RL
ai_safety | AI Safety & Alignment | {T} | what AI systems are and how they fail; specification gaming and reward hacking; robustness and distribution shift; interpretability; alignment techniques (RLHF, constitutional methods, oversight); governance, evaluation and long-term risk

# Social Sciences & Business
economics | Economics | {Q} | scarcity, choice and opportunity cost; supply and demand; elasticity and markets; market structures; national income, inflation and unemployment; growth, trade and economic policy
microeconomics | Microeconomics | {Q} | consumer choice and utility; production and costs; perfect competition; monopoly and oligopoly; game theory in markets; welfare, externalities and information economics
macroeconomics | Macroeconomics | {Q} | GDP and economic indicators; money and banking; aggregate demand and supply; fiscal and monetary policy; open-economy macroeconomics; growth models and business cycles
finance | Finance | {Q} | money, saving and interest; time value of money; stocks, bonds and valuation; risk, return and portfolios; corporate finance; derivatives and quantitative finance
accounting | Accounting | {Q} | why businesses keep records; the accounting equation and double entry; financial statements; ratios and financial analysis; cost and managerial accounting; auditing and advanced reporting
sociology | Sociology | {T} | society and culture; socialisation and identity; groups, institutions and organisations; inequality, class and stratification; social change and movements; sociological theory and research methods
political_science | Political Science | {T} | what politics and government are; political ideologies; constitutions and institutions; elections and public opinion; comparative politics; political theory and policy analysis
law | Law | {T} | rules, rights and the rule of law; sources of law and court systems; contracts and torts; criminal law and procedure; constitutional and administrative law; legal reasoning and international law
international_relations | International Relations | {T} | states and the international system; diplomacy and foreign policy; war, peace and security; international organisations; global political economy; IR theories (realism, liberalism, constructivism)
anthropology | Anthropology | {T} | what it means to be human; culture and fieldwork; kinship, ritual and belief; human evolution; archaeology and material culture; contemporary and applied anthropology
geography | Geography | {T} | maps and location; physical landscapes and climate zones; population and settlement; resources and economic geography; GIS and remote sensing; globalisation and sustainability
linguistics | Linguistics | {T} | what language is; phonetics and phonology; morphology; syntax; semantics and pragmatics; historical, socio- and computational linguistics
education | Education & Pedagogy | {T} | how people learn; lesson planning and objectives; assessment and feedback; classroom management and motivation; curriculum design and differentiation; learning sciences and education research
business | Business Management | {T} | what businesses do; organisations and management; operations and supply chains; human resources and leadership; strategy and competitive advantage; corporate governance and global business
marketing | Marketing | {T} | customers and needs; the marketing mix; market research and segmentation; branding and positioning; digital marketing and analytics; marketing strategy and consumer behaviour research
entrepreneurship | Entrepreneurship | {T} | spotting problems and opportunities; validating ideas with customers; business models and lean startup; funding and finance for startups; growth, hiring and operations; scaling, exits and innovation strategy

# Humanities & Arts
philosophy | Philosophy | {T} | asking philosophical questions; logic and argument in philosophy; epistemology; metaphysics and mind; political philosophy; philosophy of science and contemporary debates
ethics | Ethics | {T} | right, wrong and moral intuitions; consequentialism; deontology and rights; virtue ethics; applied ethics (bioethics, tech ethics); metaethics and moral uncertainty
world_literature | World Literature | {T} | stories, fables and myths; poetry and poetic devices; the novel and narrative technique; drama and tragedy; literary movements across cultures; literary theory and criticism
creative_writing | Creative Writing | {T} | sensory description and imagery; character and dialogue; plot and structure; voice, point of view and style; poetry and experimental forms; revision and publishing-quality craft
art_history | Art History | {T} | looking at art: line, colour, form; ancient and medieval art; Renaissance and Baroque; modern art movements; global and non-Western art traditions; contemporary art and art criticism
music_theory | Music Theory | {T} | rhythm, pitch and notation; scales, keys and intervals; chords and harmony; melody, counterpoint and form; orchestration and analysis; jazz, modern harmony and composition
film_studies | Film Studies | {T} | how films tell stories; shots, framing and editing; sound and mise-en-scène; genres and authorship; film history and national cinemas; film theory and criticism
religious_studies | Religious Studies | {T} | what religion is; world religions overview; sacred texts and interpretation; ritual, practice and community; religion, ethics and society; comparative and academic study of religion
classics | Classics | {T} | Greek and Roman myths; ancient history of Greece and Rome; Homer and epic poetry; Greek drama and philosophy; Latin and Greek language basics; the classical legacy and reception
archaeology | Archaeology | {T} | what archaeologists do; excavation and stratigraphy; dating methods; reconstructing ancient lives; major archaeological sites; scientific methods and heritage ethics

# World Languages
spanish | Spanish | {T} | greetings, numbers and basic vocabulary; present tense and everyday conversation; past tenses and narration; subjunctive mood; reading literature and media; advanced composition and translation
french | French | {T} | greetings, numbers and basic vocabulary; present tense and everyday conversation; passé composé and imparfait; subjunctive and complex sentences; reading literature and media; advanced composition and translation
german | German | {T} | greetings, numbers and basic vocabulary; present tense, word order and cases; past tenses and modal verbs; subjunctive and passive voice; reading literature and media; advanced composition and translation
mandarin | Mandarin Chinese | {T} | pinyin, tones and greetings; basic characters and sentence patterns; measure words and aspect particles; complex sentences and idioms (chengyu); reading news and literature; advanced composition and translation
hindi | Hindi | {T} | Devanagari script and greetings; present tense and everyday conversation; postpositions and past tenses; compound verbs and the subjunctive; reading literature and media; advanced composition and translation
"""


def parse_catalog() -> list[tuple[str, str, str, tuple[str, ...], list[str]]]:
    """Returns (key, name, category, question types, milestones) for each line."""
    out, category = [], "Other"
    for line in CATALOG.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            category = line.lstrip("# ").strip()
            continue
        key, name, types, topics = (p.strip() for p in line.split("|"))
        milestones = [t.strip() for t in topics.split(";")]
        assert len(milestones) == 6, key
        out.append((key, name, category, tuple(types.split(",")), milestones))
    return out


def expand_syllabus(milestones: list[str]) -> tuple[str, ...]:
    """Spread six milestones over 12 classes."""
    grades = []
    for i, m in enumerate(milestones):
        grades.append(f"{m}: core ideas, definitions and basic problems")
        if i == 0:
            grades.append(f"{m}: applications and multi-step problems")
        else:
            grades.append(f"{m}: harder applications, combined with {milestones[i - 1]}")
    grades[-1] = (f"{milestones[-1]}: expert-level problems, integrating everything from "
                  f"{milestones[0]} to {milestones[-2]}")
    return tuple(grades)


ICONS = {
    "statistics": "📊", "probability": "🎲", "geometry": "📐", "linear_algebra": "🧮", "calculus": "∫",
    "number_theory": "🔢", "discrete_math": "🕸️", "optimization": "🎯", "numerical_methods": "🧰",
    "astronomy": "🔭", "astrophysics": "🌌", "earth_science": "🌋", "meteorology": "⛈️", "oceanography": "🌊",
    "materials_science": "🧱", "electronics": "🔌", "quantum_computing": "⚛️", "organic_chemistry": "⚗️",
    "biochemistry": "🧫", "genetics": "🧬", "ecology": "🌿", "neuroscience": "🧠", "microbiology": "🦠",
    "botany": "🌱", "zoology": "🦁", "anatomy": "🦴", "medicine": "🩺", "nutrition": "🥗", "psychology": "🪞",
    "public_health": "🏥", "pharmacology": "💊", "mechanical_eng": "⚙️", "electrical_eng": "⚡",
    "civil_eng": "🌉", "chemical_eng": "🏭", "aerospace_eng": "🚀", "robotics": "🤖", "control_systems": "🎛️",
    "signal_processing": "📡", "environmental_eng": "♻️", "biomedical_eng": "🫀", "data_structures": "🌳",
    "algorithms": "🧭", "machine_learning": "📈", "deep_learning": "🕸", "nlp": "💬", "computer_vision": "👁️",
    "databases": "🗄️", "operating_systems": "🧠", "networking": "🌐", "cybersecurity": "🛡️", "web_dev": "🕸️",
    "software_engineering": "🏗️", "distributed_systems": "☁️", "compilers": "🔧", "data_science": "🧪",
    "cloud_computing": "⛅", "reinforcement_learning": "🕹️", "ai_safety": "🧯", "economics": "💹",
    "microeconomics": "🛒", "macroeconomics": "🏦", "finance": "💰", "accounting": "🧾", "sociology": "👥",
    "political_science": "🗳️", "law": "⚖️", "international_relations": "🌍", "anthropology": "🏺",
    "geography": "🗺️", "linguistics": "🔤", "education": "🎓", "business": "💼", "marketing": "📣",
    "entrepreneurship": "💡", "philosophy": "🤔", "ethics": "🕊️", "world_literature": "📖",
    "creative_writing": "🖋️", "art_history": "🎨", "music_theory": "🎼", "film_studies": "🎬",
    "religious_studies": "🕯️", "classics": "🏛️", "archaeology": "⛏️", "spanish": "🇪🇸", "french": "🇫🇷",
    "german": "🇩🇪", "mandarin": "🇨🇳", "hindi": "🇮🇳",
}
