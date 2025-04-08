/*
   Copyright 2024 - Théo Daron

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/



#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifndef TUNIT_H
#define TUNIT_H 42
typedef struct TestSuite testsuite_t;
typedef struct Test test_t;
/*
Definitions of available methods.

*/

testsuite_t *t_registerTestSuite(char *name);
test_t *t_addTestToSuite(testsuite_t *suite, char *name,
                         void (*test_fn)(void *));
void t_addStaticDataToTest(test_t *test, void *data);
void t_addStartUpToTest(test_t *test, void (*d)(void *));
void t_addCleanUpToTest(test_t *test, void (*d)(void *));
int t_runSuites(int argc, char **argv);

/*
Definitions of macros and constants

*/
#define C_NORM "\033[0m"
#define C_RED "\033[0;31m"
#define C_GREEN "\033[0;32m"

#define t_errf(a, op, b, fmt)                                                  \
  fprintf(stderr,                                                              \
          C_RED "FAIL> %s(%s):%d - assertion failed " fmt " %s " fmt "" C_NORM \
                "\n",                                                          \
          __FILE__, __func__, __LINE__, a, #op, b);

#define t_assert_op(a, op, b, fmt)                                             \
  if (!((a)op(b))) {                                                           \
    t_errf(a, op, b, fmt);                                                     \
    return;                                                                    \
  }

#define t_assert_str_eq(a, b)                                                  \
  if (strcmp(a, b)) {                                                          \
    t_errf(a, ==, b, "%s");                                                    \
    return;                                                                    \
  }

#define t_assert_str_neq(a, b)                                                 \
  if (!strcmp(a, b)) {                                                         \
    t_errf(a, !=, b, "%s");                                                    \
    return;                                                                    \
  }

#define t_assert_int(a, op, b) t_assert_op(a, op, b, "%d")
#define t_assert_long_int(a, op, b) t_assert_op(a, op, b, "%ld")
#define t_assert_long_long_int(a, op, b) t_assert_op(a, op, b, "%lld")
#define t_assert_char(a, op, b) t_assert_op(a, op, b, "%c")
#define t_assert_false(a) t_assert_int(a, ==, 0)
#define t_assert_double(a, op, b) t_assert_op(a, op, b, "%f")
#define t_assert_float(a, op, b) t_assert_op(a, op, b, "%f")
#define t_assert_void(a, op, b) t_assert_op((void*) a, op, (void*) b, "%p")
#define t_assert_true(a) t_assert_int(a, ==, 1)

#endif
/*
Implementation of methods.

*/
#ifdef TUNIT_IMPLEMENTATION

// Global variables
typedef struct Test test_t; // forward declaration
struct Test {
  test_t *next; // Test are all organized in linked lists.
  char *name;
  void (*test_fn)(void *);
  void (*start_up)(void *);
  void (*clean_up)(void *);
  void *static_data;
  int data_length;
};

typedef struct TestSuite testsuite_t; // forward declaration
typedef struct TestSuite {
  testsuite_t *next; // Suits are also organized in linked lists.
  test_t *first;
  test_t *last; // usefull to add test to end of the list
  char *name;
  size_t length;
} testsuite_t;

typedef struct TestSuiteList {
  testsuite_t *first;
  testsuite_t *last;
  size_t length;
} testsuitelist_t;

// This variable is global and contains the full list of all suites.
testsuitelist_t suite_list = {NULL, 0};

testsuite_t *t_registerTestSuite(char *name) {
  testsuite_t *new_suite = (testsuite_t *)malloc(sizeof(testsuite_t));
  new_suite->name = name;
  new_suite->first = NULL;
  new_suite->next = NULL;
  new_suite->last = NULL;
  suite_list.length++;

  // empty list
  if (suite_list.first == NULL) {
    suite_list.first = new_suite;
    suite_list.last = new_suite;
    return new_suite;
  }
  suite_list.last->next = new_suite;
  suite_list.last = new_suite;
  return new_suite;
}

// This method is usefull to read stdout and stderr from child processes
// used to run tests.
static char *getContent(FILE *file, int length) {
  rewind(file);
  char *output = (char *)calloc(length, 1);
  fread(output, 1, length, file);
  return output;
}

static int pv_t_runTest(test_t *test, void *input, int current_iter) {
  // when tests are ran with multiple iterations with each one their own input
  // the input is stored in curr_input
  // test->static_data contain all the inputs from all iterations
  // even if there is only one iteration (default behavior) the data will be in
  // curr_input
  void *curr_input = input;
  FILE *new_stderr = tmpfile();
  FILE *new_stdout = tmpfile();
  pid_t pid = fork();

  // here we are inside the fork
  if (pid == 0) {
    // replacing stderr and stdout by my custom ones
    dup2(fileno(new_stderr), STDERR_FILENO);
    dup2(fileno(new_stdout), STDOUT_FILENO);
    // running the test
    if (test->start_up != NULL && current_iter == 0) {
      //we only run start_up for the first iteration
      test->start_up(test->static_data);
    }
    test->test_fn(curr_input);
    if (test->clean_up != NULL && current_iter == test->data_length - 1) {
      //we only run clean_up for the end iteration
      test->clean_up(test->static_data);
    }
    exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);

  // If this length is bigger than 0 it means some errors were written
  // So test failed
  long int stderr_length = ftell(new_stderr);
  bool error = status != 0 || stderr_length > 0;

  // Reading content of test's stdout and stderr in case of error
  char *error_output = NULL;
  char *output = NULL;
  if (error) {
    int stdout_length = ftell(new_stdout);
    output = getContent(new_stdout, stdout_length);
    error_output = getContent(new_stderr, stderr_length);
  }

  // printing test name in green or red in case of success/error
  char *color = error ? C_RED : C_GREEN;
  if (current_iter == 0) {
    printf("\t-> %s%s\n" C_NORM, color, test->name);
  }
  if (test->data_length > 1) {
    printf("\t\t %s iteration=%d " C_NORM " \n", color, current_iter);
  }

  // writing stderr and stdout in case of errors
  if (output != NULL) {
    printf("%s", output);
  }
  if (error_output != NULL) {
    printf("%s", error_output);
  }

  close(fileno(new_stdout));
  close(fileno(new_stderr));
  return error ? 1 : 0;
}

static int pv_t_runIterableTest(test_t *test) {
  int res = 0;
  for (int i = 0; i < test->data_length; i++) {
    res += pv_t_runTest(test, ((char **)(test->static_data))[i], i);
  }
  return res;
}
static void pv_t_runSuite(testsuite_t *suite) {
  printf("\n---------------\n\n");
  printf("Running %s\n", suite->name);
  size_t failed = 0;
  while (suite->first != NULL) {
    test_t *test = suite->first;
    suite->first = test->next;
    int res;
    if (test->data_length > 1) {
      res = pv_t_runIterableTest(test);

    } else {
      res = pv_t_runTest(test, test->static_data, 0);
    }
    failed += res; // 1 if error and 0 if not.
    free(test);
  }
  printf("\nSucceeded " C_GREEN "%ld/%ld" C_NORM, suite->length - failed,
         suite->length);
  if (failed > 0) {
    printf(" - Failed" C_RED " %ld/%ld" C_NORM, failed, suite->length);
  }
  printf("\n");
}

void t_addStaticDataToTest(test_t *test, void *data) {
  test->static_data = data;
}

test_t *t_addTestToSuite(testsuite_t *suite, char *name,
                         void (*test_fn)(void *)) {
  test_t *t = (test_t *)malloc(sizeof(test_t));
  t->test_fn = test_fn;
  t->next = NULL;
  t->name = name;
  t->start_up = NULL;
  t->clean_up = NULL;
  t->static_data = NULL;
  t->data_length = 1;
  suite->length++;

  // empty list
  if (suite->first == NULL) {
    suite->first = t;
    suite->last = t;
    return t;
  }
  suite->last->next = t;
  suite->last = t;
  return t;
}
void t_addStartUpToTest(test_t *test, void (*startup)(void *)) {
  test->start_up = startup;
}
void t_addCleanUpToTest(test_t *test, void (*cleanup)(void *)) {
  test->clean_up = cleanup;
}

int t_runSuites(int argc, char **argv) {
  while (suite_list.first != NULL) {
    testsuite_t *suite = suite_list.first;
    if (suite->length == 0) {
      printf("Skipping %s because not tests in here\n", suite->name);
    } else {
      pv_t_runSuite(suite);
    }
    testsuite_t *next = suite->next;
    free(suite);
    suite_list.first = next;
  }
}
#endif