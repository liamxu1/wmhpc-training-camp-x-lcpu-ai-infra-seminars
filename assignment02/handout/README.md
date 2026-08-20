# handout 构建

原始材料是 `src/assignment02.md`，tex 与 pdf 由它生成，不要手工编辑
生成物。构建:`./build.sh`(依赖 pandoc、xelatex、Noto CJK 字体)。
版式与 assignment01 的 handout 一致(模板 `template.tex`)。

## md 源的约定

普通 markdown(标题、粗体、行内代码、代码块、pipe 表格、列表、$数学$、
raw latex 均可)之外,三种结构标记由 `filters/boxes.lua` 翻译:

题目标题(三级标题 + 属性;`opt`/`file` 可省略):

    ### 4.1 {.prob type=FROM-SCRATCH file=cuda/m4_gemm/01_tiled.cu}
    ### 4.4 {.prob type=FROM-SCRATCH opt=Optional}

参考资料框 / Editor's Note 框:

    ::: reading
    PTX ISA 9.7.14(mma 一节)、课件 S026-S030。
    :::

    ::: lookback
    模块收尾的评注。
    :::

压轴题框(题目整个包进框里,标题自拟):

    ::: {.capstone title="prob 3.2(FROM-SCRATCH):tcgen05 单 tile" file=cuda/m3_tcgen05/02_single_tile.cu}
    题面……
    :::

语法样例见 `src/_syntax_demo.md`,`./build.sh src/_syntax_demo.md`
可单独编译它验证工具链。
