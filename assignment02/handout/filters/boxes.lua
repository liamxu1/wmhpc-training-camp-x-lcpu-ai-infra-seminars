-- pandoc Lua filter:md 源里的结构标记 → 版式宏
--
-- 约定(见 handout/README.md):
--   ### 4.1 {.prob type=EXPERIMENT file=cuda/m4_gemm/05_thin_gemm.cu}
--     → \prob{4.1}{EXPERIMENT}[cuda/m4\_gemm/05\_thin\_gemm.cu]
--     选做加 opt=Optional;file 可省略
--   ::: reading / ::: lookback  → 对应 tcolorbox
--   ::: {.capstone title="prob 3.5(FROM-SCRATCH):block 内归约"}
--     → capstone 框

local function tex_escape(s)
  return (s:gsub("[%%#$&_{}]", "\\%0"))
end

function Header(h)
  if h.classes:includes("prob") then
    local num = pandoc.utils.stringify(h.content)
    local typ = h.attributes["type"] or "?"
    local opt = h.attributes["opt"]
    local file = h.attributes["file"]
    local cmd = "\\prob"
    if opt then cmd = cmd .. "[" .. opt .. "]" end
    cmd = cmd .. "{" .. num .. "}{" .. typ .. "}"
    if file then cmd = cmd .. "[" .. tex_escape(file) .. "]" end
    return pandoc.RawBlock("latex", cmd)
  end
end

local box_envs = { reading = true, lookback = true }

function Div(d)
  for name in pairs(box_envs) do
    if d.classes:includes(name) then
      return {
        pandoc.RawBlock("latex", "\\begin{" .. name .. "}"),
        pandoc.Div(d.content),
        pandoc.RawBlock("latex", "\\end{" .. name .. "}"),
      }
    end
  end
  if d.classes:includes("capstone") then
    local title = tex_escape(d.attributes["title"] or "")
    local file = d.attributes["file"]
    if file then
      title = title .. " \\hfill {\\footnotesize\\ttfamily " ..
        tex_escape(file) .. "}"
    end
    return {
      pandoc.RawBlock("latex", "\\begin{capstone}{" .. title .. "}"),
      pandoc.Div(d.content),
      pandoc.RawBlock("latex", "\\end{capstone}"),
    }
  end
end
