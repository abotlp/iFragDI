#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from template_mmseqs import ResolvedMmseqsHit, write_resolved_hits_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_EVIDENCE_REASON = "No sequence-backed interaction-supported homologs were found for conservation."


class ConservationAllowNoEvidenceTest(unittest.TestCase):
    def test_allow_no_evidence_writes_zero_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            query1 = tmp / "query1.fasta"
            query2 = tmp / "query2.fasta"
            query1.write_text(">Q1\nAAAA\n", encoding="utf-8")
            query2.write_text(">Q2\nCCCC\n", encoding="utf-8")

            q1_hits = tmp / "query1_hits.tsv"
            q2_hits = tmp / "query2_hits.tsv"
            write_resolved_hits_tsv(
                q1_hits,
                {
                    "P12345": ResolvedMmseqsHit(
                        accession="P12345",
                        sequence_id="P12345",
                        search_tier="template_db",
                        taxid="9606",
                        evalue=1e-20,
                        bitscore=100.0,
                        aligned_query_positions=4,
                        pident=100.0,
                        row="AAAA",
                    )
                },
            )
            write_resolved_hits_tsv(
                q2_hits,
                {
                    "Q23456": ResolvedMmseqsHit(
                        accession="Q23456",
                        sequence_id="Q23456",
                        search_tier="template_db",
                        taxid="9606",
                        evalue=1e-18,
                        bitscore=95.0,
                        aligned_query_positions=4,
                        pident=100.0,
                        row="CCCC",
                    )
                },
            )

            pairs = tmp / "pairs.tsv"
            pairs.write_text("accA\taccB\nP12345\tP99999\n", encoding="utf-8")
            pairs_meta = tmp / "pairs_meta.tsv"
            pairs_meta.write_text(
                "\t".join(
                    [
                        "accA",
                        "accB",
                        "src_intact",
                        "src_biogrid",
                        "src_string",
                        "pubmed_count",
                        "string_score_max",
                        "string_experiments_max",
                        "string_database_max",
                        "evidence_count",
                        "interaction_types",
                        "detection_methods",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "P12345",
                        "P99999",
                        "1",
                        "0",
                        "0",
                        "1",
                        "0",
                        "0",
                        "0",
                        "1",
                        "physical association",
                        "affinity chromatography",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            sequence_fasta = tmp / "sequences.fasta"
            sequence_fasta.write_text(">P12345\nAAAA\n>Q23456\nCCCC\n", encoding="utf-8")

            out_dir = tmp / "out"
            cmd = [
                sys.executable,
                "conservation.py",
                "--query1",
                str(query1),
                "--query2",
                str(query2),
                "--query1-search-tsv",
                str(q1_hits),
                "--query2-search-tsv",
                str(q2_hits),
                "--pair-dataset",
                "intact_biogrid",
                "--pairs",
                str(pairs),
                "--pairs-meta",
                str(pairs_meta),
                "--sequence-fasta",
                str(sequence_fasta),
                "--allow-no-evidence",
                "--no-heatmap",
                "--out-dir",
                str(out_dir),
            ]
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            summary = json.loads((out_dir / "conservation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "no_evidence")
            self.assertEqual(summary["pair_dataset"], "intact_biogrid")
            self.assertIsNone(summary["error"])
            self.assertEqual(summary["no_evidence_reason"], NO_EVIDENCE_REASON)
            self.assertEqual(summary["conservation_matrix_shape"], [4, 4])
            self.assertEqual(summary["paired_rows_used"], 0)
            self.assertTrue(summary["weak_msa_warning"])

            required_outputs = [
                "conservation_matrix.tsv",
                "conservation_matrix.npy",
                "conservation_freq_q1.tsv",
                "conservation_freq_q2.tsv",
                "alignment_freq_q1.tsv",
                "alignment_freq_q2.tsv",
                "query1_conservation_profile.tsv",
                "query2_conservation_profile.tsv",
            ]
            for name in required_outputs:
                self.assertTrue((out_dir / name).exists(), msg=f"missing output: {name}")

            matrix_npy = np.load(out_dir / "conservation_matrix.npy")
            self.assertEqual(matrix_npy.shape, (4, 4))
            self.assertTrue(np.allclose(matrix_npy, 0.0))

            matrix_tsv = np.loadtxt(out_dir / "conservation_matrix.tsv", delimiter="\t")
            self.assertEqual(matrix_tsv.shape, (4, 4))
            self.assertTrue(np.allclose(matrix_tsv, 0.0))

            for name in ("conservation_freq_q1.tsv", "conservation_freq_q2.tsv", "alignment_freq_q1.tsv", "alignment_freq_q2.tsv"):
                values = np.loadtxt(out_dir / name, delimiter="\t")
                self.assertEqual(values.shape, (4,))
                self.assertTrue(np.allclose(values, 0.0))

            for name, expected_aas in (("query1_conservation_profile.tsv", "AAAA"), ("query2_conservation_profile.tsv", "CCCC")):
                with (out_dir / name).open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertEqual(len(rows), len(expected_aas))
                for idx, (row, aa) in enumerate(zip(rows, expected_aas), start=1):
                    self.assertEqual(int(row["residue_index"]), idx)
                    self.assertEqual(row["aa"], aa)
                    self.assertEqual(float(row["conservation_freq"]), 0.0)
                    self.assertEqual(float(row["alignment_freq"]), 0.0)
                    self.assertEqual(float(row["profile_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
