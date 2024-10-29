def read_fasta(file_path):
    sequences = []
    current_sequence = {"id": "", "description": "", "sequence": ""}

    with open(file_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                # 处理上一个序列
                if current_sequence["id"]:
                    sequences.append(current_sequence.copy())
                    current_sequence = {"id": "", "description": "", "sequence": ""}

                # 解析新序列的标题行
                parts = line[1:].split(" ", 1)
                current_sequence["id"] = parts[0]
                current_sequence["description"] = parts[1] if len(parts) > 1 else ""
            else:
                # 累积序列数据
                current_sequence["sequence"] += line

    # 处理最后一个序列
    if current_sequence["id"]:
        sequences.append(current_sequence)

    return sequences


