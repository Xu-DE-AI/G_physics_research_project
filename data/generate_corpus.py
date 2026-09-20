import json
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TOPICS = {
    "pressure_diffusion": ["pore-pressure diffusion", "hydraulic diffusivity", "pressure front", "diffusion timescale"],
    "fault_reactivation": ["fault reactivation", "effective normal stress", "Coulomb failure", "slip tendency"],
    "injection_rate": ["injection rate", "pressure ramp", "fluid injection", "rate-dependent loading"],
    "localization": ["strain localization", "deformation localization", "fault-zone thickness", "damage localization"],
    "roughness": ["fault roughness", "asperities", "surface topography", "contact area"],
    "permeability": ["permeability evolution", "hydraulic conductivity", "fracture transmissivity", "flow localization"],
    "seismicity": ["induced seismicity", "acoustic emission", "event rate", "magnitude distribution"],
    "velocity": ["seismic velocity", "P-wave velocity", "velocity anisotropy", "wave propagation"],
}

TEMPLATES = [
    "We investigate {main} during fluid injection. The experiments indicate that {detail}. We discuss implications for {secondary}.",
    "Laboratory measurements of {main} reveal a strong relationship between {detail} and {secondary}. The observation is relevant to induced seismicity monitoring.",
    "A coupled hydro-mechanical analysis examines {main}. Results suggest that {detail}, particularly when {secondary} changes.",
    "This study focuses on {main} in a fractured sandstone. The dominant control is {detail}; {secondary} provides an additional constraint.",
]

DETAILS = {
    "pressure_diffusion": ["the pressure front propagates over a characteristic diffusion timescale", "diffusion controls the spatial extent of pore-pressure perturbations", "hydraulic diffusivity varies as fractures open"],
    "fault_reactivation": ["reduced effective normal stress promotes slip", "failure depends on both shear stress and effective normal stress", "reactivation occurs when the Coulomb criterion is approached"],
    "injection_rate": ["rapid ramps generate stronger transient pressure gradients", "loading rate changes the temporal distribution of seismic events", "the pressure ramp interacts with hydraulic storage"],
    "localization": ["deformation progressively concentrates near the principal slip surface", "localized strain is accompanied by spatially clustered events", "the width of the active deformation zone decreases during loading"],
    "roughness": ["asperity geometry controls local stress concentrations", "surface roughness modifies real contact area", "topographic mismatch influences the onset of sliding"],
    "permeability": ["permeability increases as connected fractures open", "flow becomes localized after mechanical damage", "hydraulic transmissivity evolves with shear displacement"],
    "seismicity": ["event rates respond to changes in effective stress", "magnitude distributions evolve during loading", "event locations migrate toward the active fault zone"],
    "velocity": ["wave speed changes with crack density and effective pressure", "P-wave velocity is sensitive to damage and saturation", "anisotropy develops as microcracks align"],
}

GERMAN = {
    "pressure_diffusion": "Porendruckdiffusion und hydraulische Diffusivität",
    "fault_reactivation": "Reaktivierung einer Störung und effektive Normalspannung",
    "injection_rate": "Injektionsrate und Druckrampe",
    "localization": "Deformationslokalisierung in der Störungszone",
    "roughness": "Rauheit und Asperitäten einer Störungsfläche",
    "permeability": "Entwicklung der Permeabilität",
    "seismicity": "induzierte Seismizität und akustische Emission",
    "velocity": "seismische Geschwindigkeit und Wellenausbreitung",
}


def build():
    rows = []
    idx = 0
    topics = list(TOPICS)
    for topic in topics:
        for variant in range(90):
            secondary = random.choice([t for t in topics if t != topic])
            main_term = random.choice(TOPICS[topic])
            detail = random.choice(DETAILS[topic])
            text = random.choice(TEMPLATES).format(
                main=main_term,
                detail=detail,
                secondary=random.choice(TOPICS[secondary]),
            )
            # Add controlled terminology to create realistic lexical overlap.
            text += f" Keywords: {', '.join(random.sample(TOPICS[topic], min(2, len(TOPICS[topic]))))}."
            rows.append({"doc_id": f"geo_{idx:04d}", "topic": topic, "title": f"Geophysical study: {main_term}", "text": text, "language": "en"})
            idx += 1

    # German documents create a small multilingual retrieval test.
    for topic in topics:
        for variant in range(10):
            text = (
                f"Diese Studie untersucht {GERMAN[topic]}. "
                f"Die Ergebnisse zeigen, dass {DETAILS[topic][variant % len(DETAILS[topic])]}. "
                "Die Beobachtung ist für die Überwachung induzierter Seismizität relevant."
            )
            rows.append({"doc_id": f"geo_{idx:04d}", "topic": topic, "title": f"Geophysikalische Studie: {GERMAN[topic]}", "text": text, "language": "de"})
            idx += 1

    os.makedirs("artifacts", exist_ok=True)
    pd.DataFrame(rows).to_csv("artifacts/geophysics_corpus.csv", index=False)

    queries = []
    qid = 0
    query_templates = {
        "pressure_diffusion": [
            "How does pore pressure diffuse away from an injection point?",
            "What controls the timescale of hydraulic pressure diffusion?",
            "How does hydraulic diffusivity affect induced seismicity?",
        ],
        "fault_reactivation": [
            "How does increasing pore pressure promote fault reactivation?",
            "What role does effective normal stress play in slip?",
            "How is the Coulomb failure criterion used for fault reactivation?",
        ],
        "injection_rate": [
            "How does injection rate affect induced seismicity?",
            "What happens when the pressure ramp becomes faster?",
            "How can loading rate alter the temporal distribution of events?",
        ],
        "localization": [
            "How does deformation become localized around a fault?",
            "What evidence indicates progressive strain localization?",
            "How does fault-zone thickness evolve during slip?",
        ],
        "roughness": [
            "How do fault asperities affect slip initiation?",
            "Why does surface roughness influence local stress concentration?",
            "How does real contact area affect fault stability?",
        ],
        "permeability": [
            "How does permeability evolve when fractures open?",
            "What controls hydraulic transmissivity during shear?",
            "How can flow become localized during mechanical damage?",
        ],
        "seismicity": [
            "How does induced seismicity respond to effective stress?",
            "How do event locations migrate during fluid injection?",
            "How can acoustic emission rate indicate fault evolution?",
        ],
        "velocity": [
            "How does effective pressure affect P-wave velocity?",
            "What causes seismic velocity to change during damage?",
            "How can crack alignment create velocity anisotropy?",
        ],
    }
    for topic, qs in query_templates.items():
        relevant = [r["doc_id"] for r in rows if r["topic"] == topic][:5]
        for q in qs:
            queries.append({"query_id": f"q_{qid:03d}", "query": q, "relevant_doc_ids": relevant})
            qid += 1
    # German queries for cross-lingual retrieval evaluation.
    german_queries = {
        "pressure_diffusion": "Wie beeinflusst Porendruckdiffusion die induzierte Seismizität?",
        "fault_reactivation": "Wie fördert erhöhter Porendruck die Reaktivierung einer Störung?",
        "injection_rate": "Wie beeinflusst die Injektionsrate die induzierte Seismizität?",
        "localization": "Wie lokalisiert sich die Deformation in einer Störungszone?",
    }
    for topic, q in german_queries.items():
        relevant = [r["doc_id"] for r in rows if r["topic"] == topic and r["language"] == "de"][:5]
        queries.append({"query_id": f"q_{qid:03d}", "query": q, "relevant_doc_ids": relevant, "language": "de"})
        qid += 1

    with open("artifacts/queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Created {len(rows)} documents and {len(queries)} queries.")


if __name__ == "__main__":
    build()
