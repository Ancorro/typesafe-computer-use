from dataclasses import replace
from pathlib import Path

import pytest
from conftest import busy_page
from PIL import Image, ImageDraw

from typesafe_computer_use import perception
from typesafe_computer_use.models import AxNode, Field, Item, Screen
from typesafe_computer_use.perception import (
    drawn,
    goal_echoes,
    is_echo,
    merge_blocks,
    merge_sources,
    order_items,
    to_items,
)
from typesafe_computer_use.platform_adapter import desktop


def line(text, x1, y1, x2, y2, conf=1.0):
    return (text, conf, (float(x1), float(y1), float(x2), float(y2)))


def test_merges_stacked_lines_across_columns():
    lines = [
        line("Kash Patel defends", 1080, 1531, 1300, 1561),
        line("Two House Democrats", 1400, 1531, 1600, 1561),
        line("removing bestiality as", 1075, 1571, 1300, 1601),
        line("defect again on key vote", 1400, 1571, 1600, 1601),
        line("FBI applicants", 1080, 1606, 1300, 1636),
    ]
    texts = sorted(t for t, _, _ in merge_blocks(lines))
    assert texts == ["Kash Patel defends removing bestiality as FBI applicants", "Two House Democrats defect again on key vote"]


def test_does_not_merge_far_or_misaligned_lines():
    lines = [line("Home", 100, 100, 200, 130), line("World", 400, 100, 500, 130), line("Footer", 100, 900, 200, 930)]
    assert len(merge_blocks(lines)) == 3


def test_merged_block_keeps_min_confidence_and_union_box():
    lines = [line("a", 100, 100, 200, 130, conf=1.0), line("b", 102, 140, 260, 170, conf=0.5)]
    ((text, conf, box),) = merge_blocks(lines)
    assert text == "a b" and conf == 0.5 and box == (100, 100, 260, 170)


def test_reading_order_rows_then_columns():
    lines = [line("right", 800, 100, 900, 130), line("left", 100, 105, 200, 135), line("below", 100, 300, 200, 330)]
    assert [it.text for it in to_items(lines, 255)] == ["left", "right", "below"]


def test_budget_caps_items():
    lines = [line(str(i), 100, 100 + 40 * i, 200, 130 + 40 * i) for i in range(10)]
    assert len(to_items(lines, 3)) == 3


def test_goal_echo_matches_wrapped_command_lines():
    goal = "go to cnn and click onto something related to AI on the homepage"
    echoes = goal_echoes(goal)
    assert is_echo('clear && uv run clicker "go to cnn and click onto something', echoes)
    assert is_echo('related to AI on the homepage" --act', echoes)
    assert not is_echo("Trending: Trump and AI warnings", echoes)


def ocr_item(index, text, x1, y1, x2, y2, conf=0.9):
    return Item(index, text, conf, float(x1), float(y1), float(x2), float(y2))


def ax_item(index, text, x1, y1, x2, y2, role="button"):
    return Item(index, text, 1.0, float(x1), float(y1), float(x2), float(y2), role=role, source="ax")


def test_merge_folds_an_overlapping_control_onto_the_ocr_block_that_names_it():
    block = ocr_item(0, "Register Now", 100, 100, 300, 130)
    control = ax_item(0, "Register Now for Disrupt", 110, 102, 290, 128, role="link")
    (merged,) = merge_sources([block], [control])
    assert merged.source == "ax+ocr" and merged.role == "link"
    assert merged.text == "Register Now for Disrupt"  # the longer of the two labels
    assert (merged.x1, merged.y1, merged.x2, merged.y2) == (100.0, 100.0, 300.0, 130.0)  # the OCR box


def test_merge_matches_on_shared_words_not_only_containment():
    block = ocr_item(0, "Buy tickets now", 100, 100, 300, 130)
    control = ax_item(0, "Buy tickets", 100, 100, 300, 130)
    assert [it.source for it in merge_sources([block], [control])] == ["ax+ocr"]


def test_merge_keeps_both_when_the_boxes_overlap_but_the_text_does_not_agree():
    block = ocr_item(0, "Search the docs", 100, 100, 300, 130)
    control = ax_item(0, "Clear input", 100, 100, 300, 130)
    assert sorted(it.source for it in merge_sources([block], [control])) == ["ax", "ocr"]


def test_merge_keeps_both_when_the_text_agrees_but_the_boxes_are_apart():
    block = ocr_item(0, "Share", 100, 100, 200, 130)
    control = ax_item(0, "Share", 900, 600, 960, 630)
    assert sorted(it.source for it in merge_sources([block], [control])) == ["ax", "ocr"]


def test_merge_consumes_each_ocr_block_at_most_once():
    block = ocr_item(0, "Send", 100, 100, 200, 130)
    controls = [ax_item(0, "Send", 100, 100, 200, 130), ax_item(1, "Send", 104, 104, 196, 126)]
    merged = merge_sources([block], controls)
    assert sorted(it.source for it in merged) == ["ax", "ax+ocr"]


def test_merge_lets_a_button_or_link_stand_for_the_symbol_read_off_its_icon():
    blocks = [ocr_item(0, "\u00d7", 1893, 43, 1901, 51), ocr_item(1, "▶", 1305, 458, 1309, 466)]
    controls = [ax_item(0, "Close", 1882, 27, 1920, 62), ax_item(1, "Security", 1291, 446, 1323, 479, role="link")]
    assert [(it.text, it.source) for it in merge_sources(blocks, controls)] == [("Close", "ax"), ("Security", "ax")]


def test_merge_keeps_a_symbol_on_a_tab_or_a_row_and_any_read_with_a_letter_or_digit():
    blocks = [
        ocr_item(0, "\u00d7", 300, 40, 310, 50),  # a tab's close box: a target of its own
        ocr_item(1, "+", 100, 240, 110, 250),  # on a row, which holds more than one target
        ocr_item(2, "C", 164, 84, 170, 96),  # a letter is text, even when it is an icon misread
        ocr_item(3, "25", 1300, 500, 1320, 520),  # the value a field shows
        ocr_item(4, "☆", 900, 900, 910, 910),  # on no control at all
    ]
    controls = [
        ax_item(0, "Settings", 130, 30, 320, 60, role="tab"),
        ax_item(1, "Downloads", 90, 230, 1000, 260, role="cell"),
        ax_item(2, "Reload", 147, 73, 181, 107),
        ax_item(3, "Show per page", 1290, 490, 1400, 530, role="field"),
    ]
    assert len(merge_sources(blocks, controls)) == len(blocks) + len(controls)


def test_merge_lets_a_control_take_its_label_from_its_row():
    """A Chrome settings row (chrome/030eeff7 step 4): the link is the arrow at the row's end, and
    carries the row's text, which OCR reads further left. One item, at the link."""
    text = "Site settings Controls what information sites can use and show"
    blocks = [ocr_item(0, text, 712, 511, 1225, 544)]
    controls = [ax_item(0, text, 1291, 511, 1323, 544, role="link")]
    (merged,) = merge_sources(blocks, controls)
    assert (merged.text, merged.role, merged.source) == (text, "link", "ax+ocr")
    assert (merged.x1, merged.y1, merged.x2, merged.y2) == (1291.0, 511.0, 1323.0, 544.0)  # the control's box


def test_a_dropdown_and_the_label_beside_it_are_one_item():
    """Chrome's bookmark bubble (chrome/7a5a7856 step 3). The Folder dropdown's name is drawn to its
    left and its value inside it, so the capture offered it three times, and the vote split
    0.37/0.36/0.26 between them: under 0.4, a stop. As one item it drew 0.96 on replay. 'Name' is not
    the field's label word for word, 'Bookmark name', so it stays a line of text."""
    blocks = [
        ocr_item(0, "Name", 1536, 182, 1574, 191),
        ocr_item(1, "Folder", 1536, 230, 1575, 241),
        ocr_item(2, "All Bookmarks", 1602, 230, 1696, 241),
        ocr_item(3, "Done", 1752, 294, 1788, 304),
    ]
    controls = [
        ax_item(0, "Bookmark name", 1591, 168, 1815, 206, role="field"),
        ax_item(1, "Folder", 1591, 218, 1815, 256, role="field"),
        ax_item(2, "Done", 1740, 286, 1800, 312),
    ]
    merged = {it.text: it for it in merge_sources(blocks, controls)}
    assert sorted((it.text, it.role, it.source) for it in merged.values()) == [
        ("Bookmark name", "field", "ax"),
        ("Done", "button", "ax+ocr"),
        ("Folder: All Bookmarks", "field", "ax+ocr"),
        ("Name", "", "ocr"),
    ]
    folder = merged["Folder: All Bookmarks"]
    assert (folder.x1, folder.y1, folder.x2, folder.y2) == (1591.0, 218.0, 1815.0, 256.0)


def test_text_inside_a_field_with_no_label_beside_it_stays_its_own():
    """A menu drawn over the new tab page's search box (chrome/2ad9387a step 3): the box's frame
    holds the menu's text, which is a target of its own, not the box's value."""
    blocks = [ocr_item(0, "Bookmark all tabs...", 1200, 391, 1339, 402)]
    controls = [ax_item(0, "Search Google or type a URL", 622, 376, 1368, 424, role="field")]
    assert sorted(it.text for it in merge_sources(blocks, controls)) == ["Bookmark all tabs...", "Search Google or type a URL"]


def test_a_label_read_with_other_punctuation_is_the_same_label():
    blocks = [ocr_item(0, "Translate...", 1571, 600, 1647, 610)]
    controls = [ax_item(0, "Translate…", 1515, 591, 1920, 620, role="other")]
    assert [(it.text, it.source) for it in merge_sources(blocks, controls)] == [("Translate…", "ax+ocr")]


def test_a_line_that_only_mentions_the_label_stays_a_target_of_its_own():
    blocks = [ocr_item(0, "Archive of 2019 invoices", 300, 400, 600, 420)]
    controls = [ax_item(0, "Archive", 1200, 395, 1230, 425)]
    assert sorted(it.source for it in merge_sources(blocks, controls)) == ["ax", "ocr"]


def test_the_same_label_on_other_rows_stays_apart():
    """Two 'Delete' buttons on two rows, and a 'Delete' heading on a third: three targets."""
    blocks = [ocr_item(0, "Delete", 100, 100, 160, 120)]
    controls = [ax_item(0, "Delete", 900, 200, 930, 230), ax_item(1, "Delete", 900, 300, 930, 330)]
    assert sorted(it.source for it in merge_sources(blocks, controls)) == ["ax", "ax", "ocr"]


def test_a_label_on_a_row_goes_to_the_nearest_control_that_carries_it():
    blocks = [ocr_item(0, "Delete", 700, 205, 760, 225)]
    controls = [ax_item(0, "Delete", 100, 200, 130, 230), ax_item(1, "Delete", 800, 200, 830, 230)]
    assert [(it.x1, it.source) for it in merge_sources(blocks, controls)] == [(100.0, "ax"), (800.0, "ax+ocr")]


def test_merge_numbers_everything_in_reading_order():
    blocks = [ocr_item(0, "below", 100, 300, 200, 330), ocr_item(1, "right", 800, 100, 900, 130)]
    controls = [ax_item(0, "left", 100, 105, 200, 135)]
    assert [(it.index, it.text) for it in merge_sources(blocks, controls)] == [(0, "left"), (1, "right"), (2, "below")]


def test_budget_drops_the_faintest_ocr_blocks_before_any_control():
    blocks = [ocr_item(i, f"text {i}", 100, 100 + 40 * i, 200, 130 + 40 * i, conf=0.3 + 0.1 * i) for i in range(3)]
    controls = [ax_item(0, "Send", 800, 100, 900, 130)]
    kept = merge_sources(blocks, controls, budget=2)
    assert sorted(it.text for it in kept) == ["Send", "text 2"]


def test_budget_falls_back_to_dropping_controls_when_only_controls_remain():
    controls = [ax_item(i, f"control {i}", 100, 100 + 40 * i, 200, 130 + 40 * i) for i in range(4)]
    assert len(merge_sources([], controls, budget=2)) == 2


def test_order_items_renumbers_rows_then_columns():
    items = [ocr_item(7, "right", 800, 100, 900, 130), ocr_item(2, "left", 100, 105, 200, 135)]
    assert [(it.index, it.text) for it in order_items(items)] == [(0, "left"), (1, "right")]


# ----- a control the capture does not show ---------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "osworld"


def control(x: float, y: float, w: float, h: float, label: str = "Agency") -> AxNode:
    return AxNode(role="AXComboBox", label=label, x=x, y=y, w=w, h=h, pressable=True)


def page(size=(300, 120)) -> Image.Image:
    return Image.new("RGB", size, (232, 236, 244))


def test_a_flat_box_is_not_drawn():
    assert not drawn(page(), control(20, 20, 200, 30), 1.0)


def test_an_empty_field_is_drawn_by_its_border():
    image = page()
    ImageDraw.Draw(image).rectangle((20, 20, 220, 50), outline=(40, 40, 40))
    assert drawn(image, control(20, 20, 201, 31), 1.0)


def test_a_label_inside_is_drawn():
    image = page()
    ImageDraw.Draw(image).rectangle((60, 30, 90, 40), fill=(30, 30, 30))
    assert drawn(image, control(20, 20, 200, 30), 1.0)


def test_a_neighbours_border_over_the_top_edge_does_not_count():
    """Chosen's clipped search box starts on the bottom border of the control above it."""
    image = page()
    ImageDraw.Draw(image).rectangle((10, 0, 250, 21), outline=(40, 40, 40))
    assert not drawn(image, control(20, 20, 200, 30), 1.0)


def test_the_scale_maps_points_to_capture_pixels():
    image = page((600, 240))
    ImageDraw.Draw(image).rectangle((120, 60, 180, 80), fill=(30, 30, 30))
    assert drawn(image, control(20, 20, 200, 30), 2.0)
    assert not drawn(image, control(20, 80, 200, 30), 2.0)


def test_chosens_closed_dropdown_shows_its_trigger_and_hides_its_search_box():
    """A crop of justice.gov's Forms page in OSWorld, at (780, 560) on the screen, with the frames
    Chrome reported for the Agency filter: Chosen's trigger, and the search box of its closed panel."""
    image = Image.open(FIXTURES / "chosen-dropdown-closed.png").convert("RGB")
    assert drawn(image, control(837 - 780, 668 - 560, 405, 32, "-Any-"), 1.0)
    assert not drawn(image, control(842 - 780, 697 - 560, 435, 31), 1.0)


def test_perceive_offers_a_hidden_control_off_the_list(monkeypatch):
    image = page((400, 200))
    ImageDraw.Draw(image).rectangle((20, 20, 120, 50), outline=(40, 40, 40))
    shown, hidden = control(20, 20, 101, 31, "Title"), control(20, 120, 200, 30)
    monkeypatch.setattr(desktop, "actionable_elements", lambda pid, w, h: ([shown, hidden], [], False))
    monkeypatch.setattr(perception, "ocr", lambda *args: [])
    screen = Screen(image=image, scale=1.0, app="Google Chrome", field=None, url=None, pid=7)
    items = perception.perceive(screen, 255, "goal")
    assert [it.text for it in items] == ["Title"]
    assert screen.offscreen == [hidden]


# ----- the focused field's own text ----------------------------------------------------------

# chrome/030eeff7 step 4, just after typing 'Do Not Track' into Chrome's settings search, with the
# boxes and frames that step recorded. OCR read the box's text, and the magnifier beside it, as one
# block, and the item question put 0.66 on it.
SEARCH = Field(role="AXTextField", label="Search settings", placeholder="", value="Do Not Track", x=683, y=167, w=624, h=24)


def settings_search(monkeypatch, field: Field | None) -> list[str]:
    blocks = [
        ocr_item(0, "Settings", 138, 170, 221, 192, conf=1.0),
        ocr_item(1, "O、 Do Not Track", 654, 163, 753, 186, conf=0.91),
        ocr_item(2, "+", 1312, 161, 1338, 196, conf=0.85),
        ocr_item(3, "Restore pages?", 1620, 170, 1747, 187, conf=1.0),
    ]
    nodes = [
        AxNode(role="AXTextField", label="Search settings", x=683, y=167, w=624, h=24, pressable=False),
        AxNode(role="AXButton", label="Clear search", x=1307, y=165, w=28, h=28, pressable=False),
    ]
    monkeypatch.setattr(desktop, "actionable_elements", lambda pid, w, h: (nodes, [], False))
    monkeypatch.setattr(perception, "ocr", lambda *args: blocks)
    screen = Screen(image=busy_page((1920, 1080)), scale=1.0, app="Google Chrome", field=field, url=None, pid=7)
    return [it.text for it in perception.perceive(screen, 255, "turn on Do Not Track")]


def test_the_text_in_the_focused_field_is_the_field_and_not_an_item_of_its_own(monkeypatch):
    # '+' goes too, as the icon OCR read off the 'Clear search' button, which stands for it.
    assert sorted(settings_search(monkeypatch, SEARCH)) == ["Clear search", "Restore pages?", "Search settings", "Settings"]


@pytest.mark.parametrize(
    "field",
    [None, replace(SEARCH, role="AXTextArea"), replace(SEARCH, role="AXButton", label="Ads privacy sub-page back button")],
    ids=["nothing focused", "a text area", "not a text field"],
)
def test_the_same_text_stays_when_that_field_is_not_a_focused_one_line_field(monkeypatch, field):
    assert "O、 Do Not Track" in settings_search(monkeypatch, field)
