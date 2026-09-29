"""The annotator's names for the taxonomy classes (task U2b).

``configs/taxonomy.yaml`` names classes the way the dataset needs them --
``cover``, ``psu_latch`` -- and the task card used to show exactly that.  The
second trial's annotator read "Draw cover.02 (cover) on this frame" and did not
know which part of the machine it meant.  This is the Chinese an annotator
would say instead, refined by the one attribute that changes the name
(a ``cover`` *of the CPU cooler* is the air shroud, 导风罩).

Display only: nothing is keyed by these strings, and a class this table does
not know -- a latch class added later -- is shown by its own name rather than
guessed at.
"""
from __future__ import annotations

from typing import Optional

__all__ = ["CLASS_ZH", "class_zh", "instance_label"]

CLASS_ZH: dict[str, str] = {
    "chassis": "机箱",
    "drive_cage": "硬盘支架",
    "cover": "盖板",
    "cooler_bracket": "散热器支架",
    "motherboard": "主板",
    "cpu": "CPU",
    "cpu_cooler": "CPU 散热器",
    "ram_module": "内存条",
    "psu": "电源",
    "storage_drive": "硬盘",
    "optical_drive": "光驱",
    "expansion_card": "扩展卡",
    "case_fan": "机箱风扇",
    "misc_part": "其它零件",
    "screw": "螺丝",
    "ram_latch": "内存卡扣",
    "cpu_socket_lever": "CPU 压杆",
    "psu_latch": "电源卡扣",
    "drive_latch": "硬盘卡扣",
    "card_latch": "扩展卡卡扣",
    "cooler_latch": "散热器卡扣",
    "cable_clip": "理线夹",
    "connector": "接头",
    "cable": "线缆",
}

#: ``(class, attribute) -> {value: name}`` where one attribute renames the part.
_REFINED: dict[tuple[str, str], dict[str, str]] = {
    ("cover", "of"): {
        "cpu_cooler": "导风罩", "motherboard_screws": "主板螺丝盖板",
        "ram": "内存盖板", "expansion_slot": "扩展槽挡板",
        "front_bezel": "前面板", "drive": "硬盘盖板",
    },
    ("screw", "role"): {
        "motherboard": "主板螺丝", "cpu_cooler": "散热器螺丝",
        "cooler_bracket": "散热器支架螺丝", "drive": "硬盘螺丝",
        "optical_drive": "光驱螺丝", "card": "扩展卡螺丝", "psu": "电源螺丝",
    },
    ("cpu_cooler", "kind"): {
        "fan": "CPU 风扇", "heatsink": "散热片", "heatsink_fan": "CPU 散热器",
    },
    ("storage_drive", "kind"): {"ssd": "固态硬盘", "hdd": "机械硬盘"},
    ("expansion_card", "kind"): {"gpu": "显卡", "wlan": "无线网卡"},
    ("drive_cage", "of"): {"ssd": "固态硬盘支架", "hdd": "硬盘支架",
                           "optical": "光驱支架"},
    ("connector", "kind"): {
        "atx_24pin": "主板 24pin 供电接头", "cpu_power": "CPU 供电接头",
        "sata_data": "SATA 数据接头", "sata_power": "SATA 供电接头",
        "fan": "风扇接头", "front_panel": "前面板接头", "usb_header": "USB 接头",
        "audio": "音频接头", "molex": "大 4pin 接头",
    },
}


def _class_of_key(instance: str) -> str:
    """``screw.cpu_cooler.03`` -> ``screw``: what a key says when nothing else does."""
    return str(instance).split(".", 1)[0]


def class_zh(cls: Optional[str], attrs: Optional[dict] = None,
             instance: str = "") -> str:
    """The Chinese name of a part; the class itself when there is none."""
    name = str(cls or _class_of_key(instance) or "")
    for (klass, attr), names in _REFINED.items():
        if klass == name:
            value = (attrs or {}).get(attr)
            if value is not None and str(value) in names:
                return names[str(value)]
    return CLASS_ZH.get(name, name)


def instance_label(instance: str, cls: Optional[str] = None,
                   attrs: Optional[dict] = None) -> str:
    """``导风罩 cover.01`` -- the name to read and the key to quote."""
    zh = class_zh(cls, attrs, instance)
    return f"{zh} {instance}" if zh and zh != instance else str(instance)
