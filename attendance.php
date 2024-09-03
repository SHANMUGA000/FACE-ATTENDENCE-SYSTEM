<?php
// Connect to the SQLite database (you may need to create this database)
$db = new SQLite3('attendance.db');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $selected_date = $_POST['selected_date'];

    $query = "SELECT name, time FROM attendance_data WHERE date = :selected_date";
    $stmt = $db->prepare($query);
    $stmt->bindValue(':selected_date', $selected_date, SQLITE3_TEXT);

    $result = $stmt->execute();

    $attendance_data = array();
    while ($row = $result->fetchArray(SQLITE3_ASSOC)) {
        $attendance_data[] = $row;
    }

    echo json_encode($attendance_data);
}

$db->close();
?>
