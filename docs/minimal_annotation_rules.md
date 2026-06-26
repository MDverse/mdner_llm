# Minimal annotation guidelines

## Specifications

### Entity categories & scope (The ONLY allowed categories)

You are **STRICTLY** limited to identifying and extracting entities from the following list. DO NOT use any other categories.

- **SOFTNAME**: Software used (e.g., Gromacs, AMBER, VMD)
- **SOFTVERS**: Software version (e.g., v 2016.4, 5.0.3)
- **MOL**: Names of molecules, proteins, lipids, ions, solvents (e.g., DPPC, GTP, KRas4B)
- **STIME**: Duration of the simulation (e.g., 50 ns, 200ns, 5 µs)
- **STEMP**: Temperature of the simulation (e.g., 300 K, 288K, 358K)
- **FFM**: Force fields or models (e.g., Charmm36, AMBER, MARTINI, TIP3P, OPLS)

### Strict extraction rules

- **Verbatim extraction:** Entities must be extracted **EXACTLY** as written in the input text. DO NOT merge, normalize, rephrase, or correct them in any way.
- **Precision & minimalism:** ONLY annotate the specific term. **NEVER** include descriptors (e.g., "hydrated DPPC" should be "DPPC") or vague terms (e.g., "proteins", "lipids", "sugar", "water").
- **No punctuation**: Do not include any punctuation in the extracted entities.
- **No invention:** DO NOT invent or mash up words to create entities.
