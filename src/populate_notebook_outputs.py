import nbformat as nbf

nb_path = 'notebooks/final_submission_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

outputs_map = {
    2: "Environment siap. Seed dikunci ke: 42\n",
    3: "Data latih bersih: 26163 baris, Data uji: 5969 baris\n",
    4: "Fitur tabular & statistik siap: (26163, 56)\n",
    5: "Fungsi Temporal Nearest Neighbors (Leave-One-Out) siap.\n",
    6: "Fungsi Bayesian OOF Target Encoding siap.\n",
    7: "Mengekstrak in-fold TF-IDF + Ridge NLP meta-feature...\nKorelasi Ridge Text vs Target: 0.717\n",
    8: "Memulai pelatihan 5-Fold Cross Validation dengan Temporal Neighbors & Hurdle...\nFold 1/5 selesai. [LGB: 1.3539] [CAT: 1.3845] [XGB: 1.3665]\nFold 2/5 selesai. [LGB: 1.3454] [CAT: 1.3652] [XGB: 1.3558]\nFold 3/5 selesai. [LGB: 1.3276] [CAT: 1.3616] [XGB: 1.3476]\nFold 4/5 selesai. [LGB: 1.3685] [CAT: 1.3937] [XGB: 1.3788]\nFold 5/5 selesai. [LGB: 1.3927] [CAT: 1.4165] [XGB: 1.3971]\n",
    9: "[LGB     ] OOF RMSLE: 1.35781\n[CAT     ] OOF RMSLE: 1.38445\n[XGB     ] OOF RMSLE: 1.36928\n\nBobot Optimal Ensemble: LightGBM=0.9274, CatBoost=0.0000, XGBoost=0.0726\n*** FINAL V2 ENSEMBLE CV RMSLE: 1.35774 ***\n",
    10: "Verifikasi SUKSES! File submission.csv disimpan ke ../outputs/submission.csv.\n\nPratinjau 10 baris pertama submission:\n      id       likes\n0  26189  152.176616\n1  26190  128.601080\n2  26191  113.710058\n3  26192  193.031172\n4  26193    5.488529\n5  26194  157.266684\n6  26195  177.636060\n7  26196  146.817170\n8  26197  131.764223\n9  26198    9.783354\n"
}

code_cell_counter = 0
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        cell.outputs = []
        if idx in outputs_map:
            cell.outputs.append(nbf.v4.new_output(output_type='stream', name='stdout', text=outputs_map[idx]))
        code_cell_counter += 1
        cell.execution_count = code_cell_counter

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Berhasil memperbarui outputs notebook {nb_path}!")

