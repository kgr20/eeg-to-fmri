import glob
from pathlib import Path

def check_correct_vhdr(data_dir: Path, verbose=True):
    """Check if metadata of the EEG (vhdr) is correct
    If not correct, modify it
    Args:
        data_dir (Path): directory to store EEG data. This contains data (.dat or .eeg), marker (.vmrk), and metadata (.vhrd) files
        verbose (bool): print logs or not
    """
    # list data file (.eeg or .dat)
    data_dir = Path(data_dir)  # Adding conversion to Path object
    data_path = glob.glob(str(data_dir/'*.dat')) + glob.glob(str(data_dir/'*.eeg'))

    # list marker file
    vmrk_path = glob.glob(str(data_dir/'*.vmrk'))
    # list vhdr file
    vhdr_path = glob.glob(str(data_dir/'*.vhdr'))

    assert len(data_path) == 1, f'Found multiple data files in {data_dir}'
    assert len(vmrk_path) == 1, f'Found multiple marker files in {data_dir}'
    assert len(vhdr_path) == 1, f'Found multiple vhdr files in {data_dir}'

    data_filename = Path(data_path[0]).name
    vmrk_filename = Path(vmrk_path[0]).name
    
    vhdr_path = vhdr_path[0]

    # load vhdr file
    with open(vhdr_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # flag to decide update metadata file
    is_modify = False
    for i, line in enumerate(lines):
        if line.startswith('DataFile='):
            if data_filename not in line:
                lines[i] = f'DataFile={data_filename}\n'
                is_modify = True
        elif line.startswith('MarkerFile='):
            if vmrk_filename not in line:
                lines[i] = f'MarkerFile={vmrk_filename}\n'
                is_modify = True

    if is_modify:
        # write the modified lines back to the file
        with open(vhdr_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        if verbose:
            print(f'Updated metadata to {vhdr_path}')



