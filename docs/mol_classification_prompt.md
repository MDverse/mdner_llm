# Molecule classification task

## Role definition

You classify molecular entities mentioned in descriptions of Molecular Dynamics (MD) simulation datasets.

## Categories

- **Protein, peptide**: proteins, enzymes, receptors, antibodies, peptides, protein complexes or domains, viral proteins and capsids.
- **Lipid**: lipids, fatty acids, phospholipids, sterols, lipid acronyms (POPC, DLiPC).
- **Small molecule**: drugs, ligands, inhibitors, solvents, organic compounds.
- **Ion**: ions and salts (Na+, CaCl2, ammonium salts).
- **Material, polymer**: polymers, copolymers, surfaces, crystals, zeolites, nanomaterials, hydrates.
- **PDB ID**: Protein Data Bank identifier (6LU7).
- **Sugar**: carbohydrates, mono/oligo/polysaccharides, glycans.
- **Nucleic acid**: DNA, RNA, oligonucleotides, G-quadruplexes, aptamers.
- **Sequence**: peptide, DNA or RNA sequence in one-letter code.
- **UniProt ID**: UniProt accession (P01116).
- **Undefined**: not a molecule, or not enough information.

## Rules

- Answer with exactly one category.
- Use the dataset descriptions to understand the entity: acronyms are often defined there.
- The database candidate is an entry with the same name in a public database. Say whether it is really the entity meant in the datasets: short acronyms often match unrelated entries (e.g., "AE" is the dipeptide Ala-Glu in ChEBI, but an enzyme in a dataset about phosphotriesterase variants).

## Task to execute

Classify the following entity:

- **Entity**: {{ entity }}
- **Database candidate**: {{ candidate }}

### Datasets mentioning the entity

{{ datasets }}
