(program
  (
   (comment)*
   [
     (c_style_for_statement)
     (case_statement)
     (command)
     (compound_statement)
     (declaration_command)
     (for_statement)
     (function_definition)
     (if_statement)
     (list)
     (negated_command)
     (pipeline)
     (redirected_statement)
     (subshell)
     (test_command)
     (unset_command)
     (variable_assignment)
     (variable_assignments)
     (while_statement)
   ] @statement
  )+
) @program

(function_definition
  .
  "function"
  (word)
  (compound_statement
    .
    "{"
    (
      (comment)*
      [
        (c_style_for_statement)
        (case_statement)
        (command)
        (compound_statement)
        (declaration_command)
        (for_statement)
        (function_definition)
        (if_statement)
        (list)
        (negated_command)
        (pipeline)
        (redirected_statement)
        (subshell)
        (test_command)
        (unset_command)
        (variable_assignment)
        (variable_assignments)
        (while_statement)
      ] @statement
    )+
    "}"
  )
) @function_definition

(while_statement
  condition: (
    (_) @condition
    ";"
    .
  )
) @while_statement

(do_group
  .
  "do"
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @statement
  )+
  "done"
) @do_group


(if_statement
  "if"
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @if.condition
  )+ @condition
  ";"
  "then"
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @statement
  )+
  (elif_clause)*
  (else_clause)?
  "fi" @if_statement.fi
)

(elif_clause
  "elif"
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @elif.condition
  )+ @condition
  ";"
  "then"
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @statement
  )+
)

(else_clause
  (
    (comment)*
    [
      (c_style_for_statement)
      (case_statement)
      (command)
      (compound_statement)
      (declaration_command)
      (for_statement)
      (function_definition)
      (if_statement)
      (list)
      (negated_command)
      (pipeline)
      (redirected_statement)
      (subshell)
      (test_command)
      (unset_command)
      (variable_assignment)
      (variable_assignments)
      (while_statement)
    ] @statement
  )+
)

(if_statement) @if_statement
(elif_clause) @elif_clause
(else_clause) @else_clause

("then") @then

(comment) @comment

(function_definition name: (word) @function)

(file_descriptor) @number

[
  (command_substitution)
  (process_substitution)
  (expansion)
] @embedded

(binary_expression
  operator: (_) @operator)

[
  "case"
  "in"
  "esac"
  "for"
  "while"
  "until"
  "do"
  "done"
  "export"
  "if"
  "then"
  "elif"
  "else"
  "fi"
  "function"
  "select"
  "unset"
] @keyword

(
  (command
    name: (command_name
      .
      ((word) @keyword)
      .
    )
  ) @true
  (#eq? @keyword "true")
)
(
  (command
    name: (command_name
      .
      ((word) @keyword)
      .
    )
  ) @false
  (#eq? @keyword "false")
)
(
  (command
    name: (command_name
      .
      ((word) @keyword)
      .
    )
  ) @break
  (#eq? @keyword "break")
)
(
  (
    (command
      name: (command_name
        .
        ((word) @keyword)
        .
      )
    ) @continue
  )
  (#eq? @keyword "continue")
)
(
  (
    (command
      name: (command_name
        .
        ((word) @keyword)
      )
    ) @return
  )
  (#eq? @keyword "return")
)
(
  (
    (command
      name: (command_name
        .
        ((word) @keyword)
      )
    ) @exit
  )
  (#eq? @keyword "exit")
)

(variable_name) @variable
(command_name) @command_name

[
  (string)
  (raw_string)
  (heredoc_body)
  (heredoc_start)
] @string

(if_statement
  "if"
  ([
    (c_style_for_statement)
    (case_statement)
    (command)
    (compound_statement)
    (declaration_command)
    (for_statement)
    (function_definition)
    (if_statement)
    (list)
    (negated_command)
    (pipeline)
    (redirected_statement)
    (subshell)
    (test_command)
    (unset_command)
    (variable_assignment)
    (variable_assignments)
    (while_statement)
  ]+) @if.condition.no_branch
  ";"
  "then"
  ((comment) @pragma.no_branch
    (#match? @pragma.no_branch "^# pragma: no branch"))
)

(elif_clause
  "elif"
  [
    (c_style_for_statement)
    (case_statement)
    (command)
    (compound_statement)
    (declaration_command)
    (for_statement)
    (function_definition)
    (if_statement)
    (list)
    (negated_command)
    (pipeline)
    (redirected_statement)
    (subshell)
    (test_command)
    (unset_command)
    (variable_assignment)
    (variable_assignments)
    (while_statement)
  ]+ @elif.condition.no_branch
  ";"
  "then"
  ((comment) @pragma.no_branch
    (#match? @pragma.no_branch "^# pragma: no branch"))
)

(
  (binary_expression
    operator: [
      "&&"
      "||"
    ]) @binary_expression.no_branch
  ((comment) @pragma.no_branch
    (#match? @pragma.no_branch "^# pragma: no branch"))
)


(_
  [
    (subshell)
    (redirected_statement)
    (variable_assignment)
    (variable_assignments)
    (command)
    (declaration_command)
    (unset_command)
    (test_command)
    (negated_command)
    (for_statement)
    (c_style_for_statement)
    (while_statement)
    (if_statement)
    (elif_clause)
    (else_clause)
    (case_statement)
    (pipeline)
    (list)
  ] @traced_statement
)

[
  (command)
  (compound_statement)
  (pipeline)
  (process_substitution)
  (command_substitution)
] @multiline_statements

(_) @node
