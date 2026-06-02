<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import client from '@/api/client'
import type { StrategyConfig } from '@/types/api'

// Chinese labels for rule types, allocators, and params
const RULE_CN: Record<string, string> = {
  no_filter: '无过滤（全部通过）',
  score_threshold: '分数阈值过滤',
  limit_up_filter: '涨停板过滤',
  gap_up_filter: '高开幅度过滤',
  st_filter: 'ST股过滤',
  one_day_spike: '单日异动过滤',
  volume_surge: '成交量异常过滤',
  friday_filter: '周五过滤',
  max_score_filter: '最高分限制',
  price_trailing_stop: '价格追踪止损',
}

const PARAM_CN: Record<string, string> = {
  min_score: '最低分数',
  max_score: '最高分数',
  max_gap_pct: '最大高开%',
  max_vol_ratio: '最大量比',
  max_turnover: '最大换手率',
  max_drawdown: '最大回撤(%)',
}

const ALLOCATOR_CN: Record<string, string> = {
  score_weighted: '分数加权分配',
  equal_weight: '等权分配',
}

const strategies = ref<StrategyConfig[]>([])
const drawerVisible = ref(false)
const editingName = ref<string | null>(null)
const form = ref<StrategyConfig>({
  name: '', description: '', entry_rules: [], exit_rules: [], allocator: { type: 'score_weighted', params: {} },
  max_positions: 999, max_position_pct: 1.0, initial_capital: 100000, daily_cash_pct: 0.5,
})

// Entry rules: match backend ENTRY_RULES registry
const entryRuleOptions = [
  { label: '周五过滤', value: 'friday_filter' },
  { label: 'ST股过滤', value: 'st_filter' },
  { label: '分数阈值过滤', value: 'score_threshold' },
  { label: '最高分限制', value: 'max_score_filter' },
  { label: '单日异动过滤', value: 'one_day_spike' },
  { label: '成交量异常过滤', value: 'volume_surge' },
  { label: '高开幅度过滤', value: 'gap_up_filter' },
  { label: '涨停板过滤', value: 'limit_up_filter' },
  { label: '无过滤（全部通过）', value: 'no_filter' },
]

// Exit rules: match backend EXIT_RULES registry (only price_trailing_stop)
const exitRuleOptions = [
  { label: '价格追踪止损', value: 'price_trailing_stop' },
]

const allocatorOptions = [
  { label: '分数加权分配', value: 'score_weighted' },
  { label: '等权分配', value: 'equal_weight' },
]

onMounted(loadStrategies)

async function loadStrategies() {
  try {
    const res = await client.get('/backtest/strategies')
    strategies.value = res.data?.data || res.data || []
  } catch (e) {
    console.error('Failed to load strategies', e)
  }
}

function getDefaultParams(type: string): Record<string, any> {
  const defaults: Record<string, Record<string, any>> = {
    no_filter: {},
    friday_filter: {},
    st_filter: {},
    limit_up_filter: {},
    score_threshold: { min_score: 0.5 },
    max_score_filter: { max_score: 0.7 },
    one_day_spike: {},
    volume_surge: { max_vol_ratio: 3.0, max_turnover: 0.15 },
    gap_up_filter: { max_gap_pct: 0.04 },
    price_trailing_stop: { max_drawdown: 0.10 },
  }
  return defaults[type] || {}
}

function openCreate() {
  editingName.value = null
  form.value = {
    name: '', description: '', entry_rules: [], exit_rules: [], allocator: { type: 'score_weighted', params: {} },
    max_positions: 999, max_position_pct: 1.0, initial_capital: 100000, daily_cash_pct: 0.5,
    commission_rate: 0.00025, stamp_duty_rate: 0.0005, min_commission: 5.0,
    gap_up_pct: undefined, enable_limit_up_filter: true, is_system: false,
  }
  drawerVisible.value = true
}

function openEdit(s: StrategyConfig) {
  editingName.value = s.name
  form.value = JSON.parse(JSON.stringify(s))
  drawerVisible.value = true
}

async function saveStrategy() {
  try {
    await client.post('/backtest/strategies', form.value)
    ElMessage.success(editingName.value ? '策略已更新' : '策略已创建')
    drawerVisible.value = false
    await loadStrategies()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '保存失败')
  }
}

async function deleteStrategy(name: string) {
  try {
    await ElMessageBox.confirm(`确定删除策略「${name}」？`, '确认删除', { type: 'warning' })
    await client.delete(`/backtest/strategies/${encodeURIComponent(name)}`)
    ElMessage.success('已删除')
    await loadStrategies()
  } catch (e) {
    if (e !== 'cancel') ElMessage.error('删除失败')
  }
}

function addEntryRule(type: string) {
  form.value.entry_rules.push({ type, params: getDefaultParams(type) })
}

function addExitRule(type: string) {
  form.value.exit_rules.push({ type, params: getDefaultParams(type) })
}

function removeEntryRule(idx: number) { form.value.entry_rules.splice(idx, 1) }
function removeExitRule(idx: number) { form.value.exit_rules.splice(idx, 1) }

function cnLabel(type: string): string { return RULE_CN[type] || type }
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2 class="page-title">策略配置</h2>
      <el-button type="primary" :icon="Plus" @click="openCreate">新建策略</el-button>
    </div>

    <el-row :gutter="16">
      <el-col v-for="s in strategies" :key="s.name" :span="12" style="margin-bottom: 16px">
        <el-card shadow="never" class="strategy-card">
          <div class="card-header">
            <span class="card-name">{{ s.name }}</span>
            <el-tag v-if="s.is_system" type="info" size="small">系统默认</el-tag>
          </div>
          <p class="card-desc">{{ s.description }}</p>
          <div class="card-rules">
            <span class="rule-label">入场:</span>
            <el-tag v-for="(r, i) in s.entry_rules" :key="'e'+i" size="small" type="success" style="margin:2px">{{ cnLabel(r.type) }}</el-tag>
            <span v-if="!s.entry_rules?.length" style="color: #909399">—</span>
          </div>
          <div class="card-rules">
            <span class="rule-label">卖出:</span>
            <el-tag v-for="(r, i) in s.exit_rules" :key="'x'+i" size="small" type="danger" style="margin:2px">{{ cnLabel(r.type) }}</el-tag>
            <span v-if="!s.exit_rules?.length" style="color: #909399">—</span>
          </div>
          <div class="card-rules">
            <span class="rule-label">分配:</span>
            <el-tag size="small" type="warning">{{ ALLOCATOR_CN[s.allocator?.type] || s.allocator?.type || '—' }}</el-tag>
            <span style="margin-left:8px;color:#909399;font-size:12px">
              最大持仓{{ s.max_positions }} | 单只≤{{ (s.max_position_pct * 100).toFixed(0) }}%
            </span>
          </div>
          <div class="card-actions">
            <el-button size="small" text @click="openEdit(s)">编辑</el-button>
            <el-button size="small" text type="danger" :disabled="s.is_system" @click="deleteStrategy(s.name)">删除</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Create/Edit Drawer -->
    <el-drawer v-model="drawerVisible" :title="editingName ? '编辑策略' : '新建策略'" size="520px">
      <el-form label-position="top">
        <el-form-item label="策略名称" required>
          <el-input v-model="form.name" placeholder="例如：我的策略" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" type="textarea" :rows="2" placeholder="策略描述" />
        </el-form-item>

        <el-form-item label="入场规则">
          <div style="margin-bottom:8px">
            <el-dropdown @command="addEntryRule">
              <el-button size="small">+ 添加入场规则</el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item v-for="opt in entryRuleOptions" :key="opt.value" :command="opt.value">{{ opt.label }}</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
          <div v-for="(r, i) in form.entry_rules" :key="'fe'+i" class="rule-item">
            <el-tag closable @close="removeEntryRule(i)">{{ cnLabel(r.type) }}</el-tag>
            <!-- score_threshold params -->
            <template v-if="r.type === 'score_threshold'">
              <span style="margin-left:6px;font-size:12px">{{ PARAM_CN.min_score }}:</span>
              <el-input-number v-model="r.params.min_score" :min="0" :max="1" :step="0.05" size="small" style="width:110px" />
            </template>
            <!-- max_score_filter params -->
            <template v-if="r.type === 'max_score_filter'">
              <span style="margin-left:6px;font-size:12px">{{ PARAM_CN.max_score }}:</span>
              <el-input-number v-model="r.params.max_score" :min="0" :max="1" :step="0.05" size="small" style="width:110px" />
            </template>
            <!-- gap_up_filter params -->
            <template v-if="r.type === 'gap_up_filter'">
              <span style="margin-left:6px;font-size:12px">{{ PARAM_CN.max_gap_pct }}:</span>
              <el-input-number v-model="r.params.max_gap_pct" :min="0.01" :max="0.15" :step="0.01" size="small" style="width:110px" />
            </template>
            <!-- volume_surge params -->
            <template v-if="r.type === 'volume_surge'">
              <span style="margin-left:6px;font-size:12px">{{ PARAM_CN.max_vol_ratio }}:</span>
              <el-input-number v-model="r.params.max_vol_ratio" :min="1" :max="10" :step="0.5" size="small" style="width:110px" />
              <span style="margin-left:4px;font-size:12px">{{ PARAM_CN.max_turnover }}:</span>
              <el-input-number v-model="r.params.max_turnover" :min="0.05" :max="0.5" :step="0.05" size="small" style="width:110px" />
            </template>
          </div>
        </el-form-item>

        <el-form-item label="卖出规则">
          <div style="margin-bottom:8px">
            <el-dropdown @command="addExitRule">
              <el-button size="small">+ 添加卖出规则</el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item v-for="opt in exitRuleOptions" :key="opt.value" :command="opt.value">{{ opt.label }}</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
          <div v-for="(r, i) in form.exit_rules" :key="'fx'+i" class="rule-item">
            <el-tag type="danger" closable @close="removeExitRule(i)">{{ cnLabel(r.type) }}</el-tag>
            <!-- price_trailing_stop params -->
            <template v-if="r.type === 'price_trailing_stop'">
              <span style="margin-left:6px;font-size:12px">{{ PARAM_CN.max_drawdown }}:</span>
              <el-input-number v-model="r.params.max_drawdown" :min="0.01" :max="0.5" :step="0.01" size="small" style="width:110px" />
            </template>
          </div>
        </el-form-item>

        <el-form-item label="仓位分配器">
          <el-select v-model="form.allocator.type" style="width:100%">
            <el-option v-for="opt in allocatorOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
          </el-select>
        </el-form-item>

        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="最大持仓数">
              <el-input-number v-model="form.max_positions" :min="1" :max="999" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="单只最大仓位">
              <el-input-number v-model="form.max_position_pct" :min="0.05" :max="1" :step="0.05" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="每日资金占比">
              <el-input-number v-model="form.daily_cash_pct" :min="0.01" :max="1" :step="0.01" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="12">
          <el-col :span="12">
            <el-form-item label="初始资金">
              <el-input-number v-model="form.initial_capital" :min="1000" :step="10000" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="高开过滤阈值">
              <el-input-number v-model="form.gap_up_pct" :min="0" :max="0.2" :step="0.01" style="width:100%" />
              <span style="font-size:11px;color:#909399">留空=不过滤高开</span>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="12">
          <el-col :span="8">
            <el-form-item label="佣金费率">
              <el-input-number v-model="form.commission_rate" :min="0" :max="0.01" :step="0.0001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="印花税率">
              <el-input-number v-model="form.stamp_duty_rate" :min="0" :max="0.01" :step="0.0001" :precision="4" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="最低佣金">
              <el-input-number v-model="form.min_commission" :min="0" :max="50" :step="1" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="涨停板过滤">
          <el-switch v-model="form.enable_limit_up_filter" active-text="开启" inactive-text="关闭" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="drawerVisible = false">取消</el-button>
        <el-button type="primary" @click="saveStrategy">保存</el-button>
      </template>
    </el-drawer>
  </div>
</template>

<style lang="scss" scoped>
.page-container { padding: 24px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-title { margin: 0; font-size: 20px; font-weight: 600; }
.strategy-card {
  .card-header { display: flex; align-items: center; gap: 8px; }
  .card-name { font-size: 16px; font-weight: 600; }
  .card-desc { color: #909399; font-size: 13px; margin: 8px 0; }
  .card-rules { margin: 8px 0; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .rule-label { color: #606266; font-size: 12px; min-width: 40px; }
  .card-actions { margin-top: 12px; display: flex; gap: 4px; }
}
.rule-item { display: flex; align-items: center; gap: 4px; margin: 4px 0; flex-wrap: wrap; }
</style>
