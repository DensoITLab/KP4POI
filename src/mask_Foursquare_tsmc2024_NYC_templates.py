
all_tasks = {}

# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

template['source'] = "User_{} has travel history of {} and will now visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}

template['source'] = "User_{} has previously visited {} and is now going to <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}

template['source'] = "After visiting {} , User_{} is now going to visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}

template['source'] = "User {}'s travel history includes {} , but now they will call <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}

template['source'] = "Having visited {} , User_{} is now planning to visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


template = {}

template['source'] = "User_{} has visited {} in the past and will now visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


# Extractive QA
template = {}

template['source'] = "{} were previously visited by User_{}, but they will now visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['history', 'user_id']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}

template['source'] = "User_{} will be adding <extra_id_0> to their travel list, which previously included {} ."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}

template['source'] = "User {}'s travel history comprised of {} , but they will now add <extra_id_0> to the list."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}

template['source'] = "User {}'s past travel history involves {} , but <extra_id_0> is the upcoming place they will be interested in."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template

template = {}

template['source'] = "User_{} visited {} . Now the user wants to visit <extra_id_0>."
template['target'] = "<extra_id_0> {} <extra_id_1>"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-11"] = template


all_tasks['sequential'] =  task_subgroup_2
