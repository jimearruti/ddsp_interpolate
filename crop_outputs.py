import os

import librosa
import soundfile as sf


piece_folder = "results_normalised_finetuned_30k/normalised/AuSep_1_vn_02_Sonata"
start_time = 4.291
end_time = 12.581

# piece_folder = "results_normalised_finetuned_15k/normalised/AuSep_1_vn_35_Rondeau"
# start_time = 11.801
# end_time = 18.635

# piece_folder = "results_normalised_finetuned_15k/normalised/AuSep_2_vn_09_Jesus"
# start_time = 05.000
# end_time = 12.819



os.makedirs(os.path.join(piece_folder, "cropped"), exist_ok=True)
files_in_folder = [f for f in os.listdir(piece_folder) if f.endswith(".wav")]

for file_path in files_in_folder:
    full_path = os.path.join(piece_folder, file_path)
    y, sr = librosa.load(full_path)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    y_cropped = y[start_sample:end_sample]
    filename = os.path.splitext(file_path)[0]
    out_path = os.path.join(piece_folder, "cropped", f"{filename}_cropped_{int(start_time * 1000)}_{int(end_time * 1000)}.wav")
    sf.write(out_path, y_cropped, sr)