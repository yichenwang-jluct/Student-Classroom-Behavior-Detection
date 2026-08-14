# Split lists

`train.txt`, `val.txt` and `test.txt` contain the **filenames** of the images in
each partition — 2,061 / 197 / 98 lines respectively. No images are included.

Generate them from a local copy of the dataset with:

    python - <<'PY'
    import os
    for src, dst in [('train', 'train.txt'), ('valid', 'val.txt'), ('test', 'test.txt')]:
        d = os.path.join('..', 'datasets', src, 'images')
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(os.listdir(d))) + '\n')
    PY

Check the filenames before publishing: remove anything that identifies a room,
a date or a person.
