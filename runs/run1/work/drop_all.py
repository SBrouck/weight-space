from kernel import current_edit, drop_edit
for n in current_edit():
    drop_edit(n)
print('remaining edit:', current_edit())
