# coding=utf-8
import sys, copy
import cPickle as pickle

if __name__ == '__main__':
    infile = sys.argv[1]
    seqFile = sys.argv[2]
    typeFile = sys.argv[3]
    outFile = sys.argv[4]

    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0
    # 对于文件中的每一行 （表示一个分类信息）
    for line in infd:
        # 读取icd到类别的映射
        tokens = line.strip().split(',')
        # 当前的icd码
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4:
                icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3:
                icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types:
            missList.append(icd9)
        else:
            hitList.append(icd9)

        # 首先给每个CCS类别都分配一个type编号
        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types:
                cat4count += 1
                types[desc4] = len(types)
    infd.close()
    rTypes = dict([(v, k) for k, v in types.iteritems()])
    rootCode = len(types)
    types['A_ROOT'] = rootCode
    print rootCode  # 总type的数量

    pickle.dump(types, open(outFile + '.ancestors.types', 'wb'), -1)

    # CCS各个级别类别的计数
    print 'cat1count: %d' % cat1count
    print 'cat2count: %d' % cat2count
    print 'cat3count: %d' % cat3count
    print 'cat4count: %d' % cat4count
    print 'Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1)
    # print 'hit count: %d' % len(set(hitList))
    print 'miss count: %d' % len(startSet - set(hitList))
    # 这里的missSet表示没有类别信息（不存在ccs分类中）的疾病
    # 前面的missList表示存在于CCS但不存在于MIMIC数据集中的ICD疾病
    missSet = startSet - set(hitList)

    # pickle.dump(types, open(outFile + '.types', 'wb'), -1)
    # pickle.dump(missSet, open(outFile + '.miss', 'wb'), -1)

    # 上面是为了给所有CCS类别分配一个type号，这里重新扫描CCS分类文件来完成映射
    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    # 先给miss的icd码分配一个type号，直接放在rootCode下
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue
        # 获取对应icd9编码的type code
        icdCode = types[icd9]

        codeVec = []
        # 分别存储每个icd码对应的类别信息
        # 尽量存储更详细的
        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]

    # Now we re-map the integers to all medical codes.
    # 将CCS类别映射到icd code列表
    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    # rtypes是types的反映射：type_num->icd_code
    rtypes = dict([(v, k) for k, v in types.iteritems()])

    # 下述代码构建一套新的type:newTypes
    # 这里newTypes用于与其他的newMap一起，来记录每个icd码的ancestors
    codeCount = 0
    # 这里fiveMap是icd的type_num到对应类别信息ancestors的列表的映射
    for icdCode, ancestors in fiveMap.iteritems():
        # 在newTypes中添加 icd_code --> newTypeNum 的映射
        newTypes[rtypes[icdCode]] = codeCount
        # 在newFiveMap中添加newTypeNum --> [newTypeNum, ancestors(包括rootCode, code1, code2, code3, code4)]的映射
        # 但是注意这里的code1、code2等仍然是服从于type字典
        newFiveMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in fourMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in threeMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in twoMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in oneMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                # 为每一个code添加newTypeNum
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
