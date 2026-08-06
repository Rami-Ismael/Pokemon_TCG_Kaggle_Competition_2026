"""Public-board, handwritten tactical guards.

Each rule is a narrow, deterministic if/then condition.  No model, fitted
parameter, replay hash, serialized table, or search is consulted.
"""
from .boss_damaged_basic_stall_guard import choose as boss_damaged_basic
from .boss_unenergized_kadabra_stall_guard import choose as boss_kadabra
from .damaged_munkidori_early_promotion_guard import choose as damaged_munk_promotion
from .dead_poffin_guard import choose as dead_poffin
from .energized_impidimp_preservation_guard import choose as preserve_imp
from .energized_mixed_basic_promotion_guard import choose as mixed_basic_promotion
from .energized_morgrem_preservation_guard import choose as preserve_morgrem
from .energized_munkidori_promotion_guard import choose as energized_munk_promotion
from .low_hp_mega_munkidori_promotion_guard import choose as low_hp_mega_munk
from .munkidori_lethal_guard import choose as munk_lethal
from .preability_active_impidimp_attachment_guard import choose as preability_imp_attach
from .preability_morgrem_attachment_guard import choose as preability_morg_attach
from .preattack_froslass_attachment_guard import choose as preattack_froslass_attach
from .prestamp_attachment_guard import choose as prestamp_attach
from .punkup_backup_guard import choose as punkup_backup
from .rare_candy_impidimp_promotion_guard import choose as candy_imp_promotion
from .retreat_impidimp_preserve_grimmsnarl_guard import choose as retreat_imp_preserve
from .retreat_morgrem_stall_guard import choose as retreat_morgrem
from .sacrificial_froslass_mega_promotion_guard import choose as sacrificial_froslass
from .shadow_bullet_double_ko_guard import choose as shadow_double_ko
from .shadow_bullet_immediate_mega_ko_guard import choose as shadow_immediate_mega
from .zoroark_guard import choose as zoroark_end
from .zoroark_pokepad_guard import choose as zoroark_pad

GUARDS = (
    retreat_imp_preserve,
    sacrificial_froslass,
    low_hp_mega_munk,
    preability_imp_attach,
    retreat_morgrem,
    candy_imp_promotion,
    damaged_munk_promotion,
    preattack_froslass_attach,
    preability_morg_attach,
    boss_kadabra,
    mixed_basic_promotion,
    preserve_morgrem,
    preserve_imp,
    energized_munk_promotion,
    shadow_immediate_mega,
    boss_damaged_basic,
    shadow_double_ko,
    prestamp_attach,
    munk_lethal,
    punkup_backup,
    dead_poffin,
    zoroark_pad,
    zoroark_end,
)

def apply(obs):
    for guard in GUARDS:
        try:
            action, _ = guard(obs)
        except Exception:
            continue
        if action is not None:
            return list(action)
    return None
