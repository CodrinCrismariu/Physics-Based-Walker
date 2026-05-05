"""Repo-local Unitree G1 robot config with rubber hands removed."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import EntityCfg


_HAND_MESH_NAMES = {"left_rubber_hand", "right_rubber_hand"}
_HAND_SITE_NAMES = {"left_palm", "right_palm"}
_HAND_COLLISION_NAMES = {"left_hand_collision", "right_hand_collision"}


def _remove_hand_elements(xml_text: str) -> str:
  """Remove hand mesh assets, palm sites, and hand collision/visual geoms."""
  root = ET.fromstring(xml_text)

  asset = root.find("asset")
  if asset is not None:
    for mesh in list(asset):
      if mesh.tag == "mesh" and mesh.attrib.get("name") in _HAND_MESH_NAMES:
        asset.remove(mesh)

  for parent in root.iter():
    for child in list(parent):
      if child.tag == "site" and child.attrib.get("name") in _HAND_SITE_NAMES:
        parent.remove(child)
        continue
      if child.tag != "geom":
        continue
      if (
        child.attrib.get("mesh") in _HAND_MESH_NAMES
        or child.attrib.get("name") in _HAND_COLLISION_NAMES
      ):
        parent.remove(child)

  return ET.tostring(root, encoding="unicode")


def get_g1_no_hands_spec() -> mujoco.MjSpec:
  """Load the upstream G1 XML with the physical rubber hands removed."""
  xml_text = _remove_hand_elements(g1_constants.G1_XML.read_text())
  spec = mujoco.MjSpec.from_string(xml_text)
  spec.assets = g1_constants.get_assets(spec.meshdir)
  return spec


def get_g1_no_hands_robot_cfg() -> EntityCfg:
  """Return a fresh G1 robot config that uses the no-hands MJCF spec."""
  cfg = g1_constants.get_g1_robot_cfg()
  cfg.spec_fn = get_g1_no_hands_spec
  return cfg
