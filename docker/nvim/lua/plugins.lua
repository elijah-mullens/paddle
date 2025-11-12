return {
  -- Core UX
  { "nvim-lua/plenary.nvim" },
  { "nvim-telescope/telescope.nvim", cmd = "Telescope" },

  -- Syntax & TS
  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate",
    config = function()
      require("nvim-treesitter.configs").setup {
        ensure_installed = { "lua", "vim", "python", "cpp", "cuda", "cmake", "markdown" },
        highlight = { enable = true },
        indent = { enable = true },
      }
    end
  },

  -- Statusline
  {
    "nvim-lualine/lualine.nvim",
    config = function()
      require("lualine").setup { options = { theme = "auto" } }
    end
  },

  -- Git goodies
  {
    "lewis6991/gitsigns.nvim",
    config = function() require("gitsigns").setup() end
  },

  -- LSP, Mason, completion
  { "williamboman/mason.nvim",
    config = function() require("mason").setup() end
  },
  { "williamboman/mason-lspconfig.nvim",
    dependencies = { "neovim/nvim-lspconfig" },
    config = function()
      require("mason-lspconfig").setup {
        ensure_installed = { "clangd", "pyright" }
      }
      local lspconfig = require("lspconfig")
      lspconfig.clangd.setup {}
      lspconfig.pyright.setup {}
    end
  },
  {
    "hrsh7th/nvim-cmp",
    dependencies = {
      "hrsh7th/cmp-nvim-lsp",
      "hrsh7th/cmp-buffer",
      "hrsh7th/cmp-path",
      "L3MON4D3/LuaSnip",
    },
    config = function()
      local cmp = require("cmp")
      cmp.setup({
        snippet = { expand = function(args) require("luasnip").lsp_expand(args.body) end },
        mapping = cmp.mapping.preset.insert({
          ["<C-Space>"] = cmp.mapping.complete(),
          ["<CR>"]      = cmp.mapping.confirm({ select = true }),
          ["<C-e>"]     = cmp.mapping.abort(),
        }),
        sources = cmp.config.sources({
          { name = "nvim_lsp" }, { name = "path" }, { name = "buffer" }
        }),
      })
      -- LSP capabilities for completion
      local caps = require("cmp_nvim_lsp").default_capabilities()
      require("lspconfig").clangd.setup { capabilities = caps }
      require("lspconfig").pyright.setup { capabilities = caps }
    end
  },

  -- Formatting/Linting (optional)
  {
    "nvimtools/none-ls.nvim",
    config = function()
      local null_ls = require("null-ls")
      null_ls.setup({
        sources = {
          null_ls.builtins.formatting.clang_format,
          null_ls.builtins.formatting.black,
        },
      })
    end
  },
}
