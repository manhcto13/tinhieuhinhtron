# tinhieuhinhtron
// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © AlgoAlpha
//@version=6
indicator("Machine Learning Key Levels [AlgoAlpha]", "AlgoAlpha – ML Key Levels", overlay = true, max_bars_back = 5000, max_labels_count = 500, max_lines_count = 500)
//== INIT
type store
    float v
    int   b
//== INPUTS
clusterGR = "Clustering"
len         = input.int(14,  "Pivot Length", minval = 1, tooltip = "Number of bars to look back/forward for pivot detection", group = clusterGR)
memory      = input.int(50,  "Pivot Memory", minval = 2, tooltip = "Maximum number of pivots to store for clustering analysis, increasing too much will cause the indicator to crash.", group = clusterGR)
numClusters = input.int(5,   "Number of Levels", minval = 1, tooltip = "Fixed number of key levels to display (when Auto is disabled)", group = clusterGR)
autoLevels  = input.bool(true, "Auto No. Levels", tooltip = "Automatically determine optimal number of levels using silhouette method", group = clusterGR)
displayGR = "Display"
showPivots  = input.bool(true, "Show Pivot Circles", tooltip = "Display pivot point markers on chart", group = displayGR)
glow  = input.bool(true, "Level Glow", tooltip = "Enable/Disable cluster levels glow effect", group = displayGR)
displayMode = input.string("Percent", "Display Strength As", options = ["Percent", "Count"], tooltip = "Show level strength as percentage or raw count", group = displayGR)
paletteGR = "Color Palette"
col1 = input.color(color.yellow, "1", inline = "pal1", group = paletteGR)
col2 = input.color(color.blue, "2", inline = "pal1", group = paletteGR)
col3 = input.color(color.orange, "3", inline = "pal1", group = paletteGR)
col4 = input.color(color.red, "4", inline = "pal1", group = paletteGR)
col5 = input.color(color.teal, "5", inline = "pal2", group = paletteGR)
col6 = input.color(#26a69a, "6", inline = "pal2", group = paletteGR)
col7 = input.color(#ef5350, "7", inline = "pal2", group = paletteGR)
col8 = input.color(#787b86, "8", inline = "pal2", group = paletteGR)
metricsGR = "Metrics Table"
showMetrics = input.bool(true, "Show Metrics Table", group = metricsGR, tooltip = "Display table with ML clustering metrics")
metricsSizeStr = input.string("Small", "Table Text Size", options = ["Tiny", "Small", "Normal"], group = metricsGR)
metricsTextSize = metricsSizeStr == "Small" ? size.small : metricsSizeStr == "Normal" ? size.normal : size.tiny
metricsPos = input.string("Top Right", "Table Position", options = ["Top Left", "Top Center", "Top Right", "Middle Right", "Bottom Left", "Bottom Center", "Bottom Right"], group = metricsGR)
bool showPercent = displayMode == "Percent"
if memory < numClusters and not autoLevels
    runtime.error("Pivot Memory must be ≥ Number of Clusters")
if memory < 2 and autoLevels
    runtime.error("Pivot Memory must be ≥ 2 for Auto No. Levels")
var store[] P = array.new<store>()
var color[] pal = array.from(col1, col2, col3, col4, col5, col6, col7, col8)
//== FUNCTIONS
average_linkage_with_labels(store[] src, int k) =>
    int n = array.size(src)
    float[] means = array.new_float()
    int[]   sizes = array.new_int()
    int[]   lbl   = array.new_int(n, 0)
    for i = 0 to n - 1
        array.push(means, array.get(src, i).v)
        array.push(sizes, 1)
        array.set(lbl, i, i)
    matrix<float> D = matrix.new<float>(n, n, 0.0)
    for i = 0 to n - 1
        for j = 0 to n - 1
            matrix.set(D, i, j, math.abs(array.get(means, i) - array.get(means, j)))
    int active = n
    while active > k
        float best = na
        int   bi = na
        int   bj = na
        for i = 0 to active - 2
            for j = i + 1 to active - 1
                float d = matrix.get(D, i, j)
                if na(best) or d < best
                    best := d, bi := i, bj := j
        float mi = array.get(means, bi), mj = array.get(means, bj)
        int   si = array.get(sizes, bi), sj = array.get(sizes, bj)
        float newMean = (mi * si + mj * sj) / (si + sj)
        array.set(means, bi, newMean)
        array.set(sizes, bi, si + sj)
        for p = 0 to n - 1
            int lp = array.get(lbl, p)
            if lp == bj
                array.set(lbl, p, bi)
            else if lp > bj
                array.set(lbl, p, lp - 1)
        for c = 0 to active - 1
            if c != bi and c != bj
                float di = matrix.get(D, bi, c)
                float dj = matrix.get(D, bj, c)
                float newD = (si * di + sj * dj) / (si + sj)
                matrix.set(D, bi, c, newD)
                matrix.set(D, c, bi, newD)
        array.remove(means, bj)
        array.remove(sizes, bj)
        matrix.remove_row(D, bj)
        matrix.remove_col(D, bj)
        active -= 1
    float[] mSorted = array.copy(means)
    array.sort(mSorted, order.ascending)
    int[] map = array.new_int(active, 0)
    for i = 0 to active - 1
        array.set(map, i, array.indexof(mSorted, array.get(means, i)))
    for p = 0 to n - 1
        array.set(lbl, p, array.get(map, array.get(lbl, p)))
    [mSorted, lbl]
build_dist_matrix(store[] data) =>
    int n = array.size(data)
    matrix<float> M = matrix.new<float>(n, n, 0.0)
    for i = 0 to n - 2
        for j = i + 1 to n - 1
            float d = math.abs(array.get(data, i).v - array.get(data, j).v)
            matrix.set(M, i, j, d)
            matrix.set(M, j, i, d)
    M
calc_silhouette_score(store[] data, int[] labels) =>
    int n = array.size(data)
    matrix<float> dist = build_dist_matrix(data)
    
    // Get unique cluster labels
    int[] uniqueLabels = array.new_int()
    for i = 0 to n - 1
        int lbl = array.get(labels, i)
        if array.indexof(uniqueLabels, lbl) == -1
            array.push(uniqueLabels, lbl)
    
    float total = 0.0
    for p = 0 to n - 1
        int lbl_p = array.get(labels, p)
        
        // Calculate a_i (average distance within cluster)
        float sum_intra = 0.0
        int cnt_intra = 0
        for j = 0 to n - 1
            if j != p and array.get(labels, j) == lbl_p
                sum_intra += matrix.get(dist, p, j)
                cnt_intra += 1
        float a_i = cnt_intra > 0 ? sum_intra / cnt_intra : na
        
        // Calculate b_i (minimum average distance to other clusters)
        float b_i = na
        for cl_idx = 0 to array.size(uniqueLabels) - 1
            int cluster_lbl = array.get(uniqueLabels, cl_idx)
            if cluster_lbl != lbl_p
                float sum_other = 0.0
                int cnt_other = 0
                for j = 0 to n - 1
                    if array.get(labels, j) == cluster_lbl
                        sum_other += matrix.get(dist, p, j)
                        cnt_other += 1
                float avg_other = cnt_other > 0 ? sum_other / cnt_other : na
                if not na(avg_other) and (na(b_i) or avg_other < b_i)
                    b_i := avg_other
        
        // Calculate silhouette score for this point
        float s_i = 0.0
        if cnt_intra == 0
            s_i := 0.0
        else if a_i == 0 and b_i == 0
            s_i := 0.0
        else
            s_i := (b_i - a_i) / math.max(a_i, b_i)
        total += s_i
    
    n > 0 ? total / n : 0.0
find_optimal_clusters(store[] data, int min_k, int max_k) =>
    float best = -1.0
    int   optk = min_k
    for k = min_k to max_k
        if k >= array.size(data)
            break
        [keys_tmp, lbls_tmp] = average_linkage_with_labels(data, k)
        float score = calc_silhouette_score(data, lbls_tmp)
        if score > best
            best := score
            optk := k
    optk
getColor(int id) => array.get(pal, id % array.size(pal))
posFromString(str) =>
    switch str
        "Top Left"      => position.top_left
        "Top Center"    => position.top_center
        "Top Right"     => position.top_right
        "Middle Right"  => position.middle_right
        "Bottom Left"   => position.bottom_left
        "Bottom Center" => position.bottom_center
        "Bottom Right"  => position.bottom_right
//== CALCULATIONS
ph = ta.pivothigh(len, len)
pl = ta.pivotlow(len, len)
pv = not na(ph) ? ph : pl
if not na(pv)
    array.unshift(P, store.new(pv, bar_index - len))
while array.size(P) > memory
    array.pop(P)
float[] keys = array.new_float()
int[]   labs = array.new_int()
int effectiveClusters = numClusters
if array.size(P) >= 2 and barstate.islast
    if autoLevels
        int max_search = math.min(10, array.size(P))
        effectiveClusters := find_optimal_clusters(P, 2, max_search)
    if array.size(P) >= effectiveClusters
        [keysTmp, labsTmp] = average_linkage_with_labels(P, effectiveClusters)
        keys := keysTmp
        labs := labsTmp
//== VISUALS
earliest = array.size(P) > 0 ? array.get(P, array.size(P) - 1).b : bar_index
var line[] lines = array.new<line>()
while array.size(lines) > 0
    line.delete(array.pop(lines))
if keys.size() > 0 and barstate.islast
    for i = 0 to array.size(keys) - 1
        float level = array.get(keys, i)
        int _baseTrans = 90
        int _baseWidth = 15
        if glow
            for a = 0 to 2
                line ln_a = line.new(earliest, level, bar_index, level, xloc = xloc.bar_index,
                                     color = color.new(getColor(i), _baseTrans + a * 3),
                                     width = _baseWidth + a * 15)
                array.push(lines, ln_a)
        line ln = line.new(earliest, level, bar_index, level, xloc = xloc.bar_index, color = getColor(i), width = 2)
        array.push(lines, ln)
var label[] marks = array.new<label>()
while array.size(marks) > 0
    label.delete(array.pop(marks))
if P.size() > 0 and barstate.islast and showPivots
    for i = 0 to array.size(P) - 1
        store st = array.get(P, i)
        int cid  = i < array.size(labs) ? array.get(labs, i) : 0
        label lb = label.new(st.b, st.v, "", xloc = xloc.bar_index, yloc = yloc.price, style = label.style_circle, size = size.tiny, color = color.new(getColor(cid), 70))
        array.unshift(marks, lb)
var label[] lvlLabels = array.new<label>()
while array.size(lvlLabels) > 0
    label.delete(array.pop(lvlLabels))
if keys.size() > 0 and barstate.islast
    int[] counts = array.new_int(array.size(keys), 0)
    for i_idx = 0 to array.size(labs) - 1
        int cluster_id = array.get(labs, i_idx)
        array.set(counts, cluster_id, array.get(counts, cluster_id) + 1)
    int totalPivots = array.size(P)
    for i_lvl = 0 to array.size(keys) - 1
        float levelPrice = array.get(keys, i_lvl)
        int   cnt        = array.get(counts, i_lvl)
        float strength   = totalPivots > 0 ? float(cnt) / float(totalPivots) : 0.0
        int   transp     = math.round((0.9 - strength) * 100)
        string txt       = showPercent ? str.tostring(strength * 100, format.percent) : str.tostring(cnt)
        label lbLevel = label.new(bar_index, levelPrice, txt, xloc = xloc.bar_index, yloc = yloc.price, style = label.style_label_left, size = size.normal, color = color.new(getColor(i_lvl), transp), textcolor = chart.fg_color)
        array.push(lvlLabels, lbLevel)
if barstate.islast and showMetrics
    int pivotCount = array.size(P)
    float silScore = array.size(labs) > 0 ? calc_silhouette_score(P, labs) : na
    int[] sizeArr = array.new_int(effectiveClusters, 0)
    for l = 0 to array.size(labs) - 1
        int cid = array.get(labs, l)
        array.set(sizeArr, cid, array.get(sizeArr, cid) + 1)
    int largestSz = array.size(sizeArr) > 0 ? array.max(sizeArr) : na
    int smallestSz = array.size(sizeArr) > 0 ? array.min(sizeArr) : na
    var table mlTable = table.new(posFromString(metricsPos), 2, 6, bgcolor = color.new(#181924, 0), border_width = 1, border_color = color.rgb(0, 0, 0), frame_color = color.rgb(0, 0, 0), frame_width = 1)
    table.cell(mlTable, 0, 0, "Statistic", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 0, "Value", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 0, 1, "Pivot Count", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 1, str.tostring(pivotCount), text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 0, 2, "Cluster Count", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 2, str.tostring(effectiveClusters), text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 0, 3, "Silhouette Score", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 3, str.tostring(silScore, "#.###"), text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 0, 4, "Largest Cluster Size", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 4, str.tostring(largestSz), text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 0, 5, "Smallest Cluster Size", text_color = color.white, text_size = metricsTextSize)
    table.cell(mlTable, 1, 5, str.tostring(smallestSz), text_color = color.white, text_size = metricsTextSize)
